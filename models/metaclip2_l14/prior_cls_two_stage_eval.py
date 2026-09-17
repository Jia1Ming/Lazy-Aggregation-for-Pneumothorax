import json, argparse, random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import gaussian_kde

from transformers import MetaClip2Model, AutoImageProcessor


# config
MODEL_ID = "facebook/metaclip-2-worldwide-l14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR  = ""
LESION_DIR = ""
LUNG_DIR   = ""
OUT_BASE   = str(Path(__file__).resolve().parent)

if not (IMAGE_DIR and LESION_DIR and LUNG_DIR):
    raise SystemExit("Set IMAGE_DIR / LESION_DIR / LUNG_DIR at the top of this file.")


IMAGE_SIZE = 224
PATCH_SIZE = 14
NUM_PATCHES_1D = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = NUM_PATCHES_1D ** 2
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

MASKING_POINTS = 11
MASKING_RANDOM_TRIALS = 3
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

Path(OUT_BASE).mkdir(parents=True, exist_ok=True)
print("Device:", DEVICE)
print("Patch grid:", NUM_PATCHES_1D, "x", NUM_PATCHES_1D, "=", NUM_PATCHES)


# model
_PROCESSOR = None

def load_model_and_processor():
    global _PROCESSOR
    _PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_ID)
    trunk = MetaClip2Model.from_pretrained(MODEL_ID).vision_model.to(DEVICE).eval()
    print("Loaded:", MODEL_ID, "| trunk:", type(trunk).__name__)
    return trunk, _PROCESSOR

def preprocess(image):
    return _PROCESSOR(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]

def _run_full_tokens(trunk, pixel_values):
    h = trunk(pixel_values).last_hidden_state
    h = trunk.post_layernorm(h)
    return h

def get_last_hidden(trunk, pixel_values):
    pixel_values = pixel_values.to(DEVICE, dtype=next(trunk.parameters()).dtype)
    with torch.no_grad():
        hidden = _run_full_tokens(trunk, pixel_values)
    return hidden.float().cpu()[0]

def split_cls_patch(hidden):
    return hidden[0], hidden[1:1 + NUM_PATCHES]

def extract_whole_features(trunk, image):
    hidden = get_last_hidden(trunk, preprocess(image))
    return split_cls_patch(hidden)

def cls_from_pixel_values(trunk, pixel_values):
    return get_last_hidden(trunk, pixel_values)[0]


def forward_features_keep_from_pv(trunk, pixel_values, keep_idx):
    pixel_values = pixel_values.to(DEVICE, dtype=next(trunk.parameters()).dtype)
    keep = [0] + [int(i) + 1 for i in keep_idx]
    with torch.no_grad():
        h = trunk.embeddings(pixel_values)
        h = trunk.pre_layrnorm(h)
        h = h[:, keep, :]
        h = trunk.encoder(inputs_embeds=h)[0]
        h = trunk.post_layernorm(h)
    return h.float().cpu()[0]

def extract_lung_only_features(trunk, image, keep_idx):
    hidden = forward_features_keep_from_pv(trunk, preprocess(image), keep_idx)
    return hidden[0], hidden[1:]

def cls_lung_from_pixel_values(trunk, pixel_values, keep_idx):
    return forward_features_keep_from_pv(trunk, pixel_values, keep_idx)[0]


# masks
def resolve_mask_path(mask_dir, stem, suffix=""):
    if mask_dir is None:
        return None
    for ext in SUPPORTED_EXTS:
        p = Path(mask_dir) / f"{stem}{suffix}{ext}"
        if p.exists():
            return p
    return None

def patch_coverage_from_array(mask_array):
    mm = (mask_array > 0).astype(np.uint8) * 255
    small = Image.fromarray(mm).resize((NUM_PATCHES_1D, NUM_PATCHES_1D), Image.BILINEAR)
    return (np.asarray(small, dtype=np.float32) / 255.0).flatten()

def patch_coverage(mask_path):
    mm = np.asarray(Image.open(mask_path).convert("L")) > 127
    return patch_coverage_from_array(mm)

def make_pleural_band_coverage(lung_mask_path, band_px=18):
    lung = np.asarray(Image.open(lung_mask_path).convert("L")) > 127
    outer = binary_dilation(lung, iterations=band_px)
    inner = binary_erosion(lung, iterations=band_px)
    band = outer ^ inner
    return patch_coverage_from_array(band)

def build_dataset_index(image_dir, lung_dir, lesion_dir):
    rows = []
    for p in sorted(Path(image_dir).iterdir()):
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        lung = resolve_mask_path(lung_dir, p.stem)
        lesion = resolve_mask_path(lesion_dir, p.stem)
        if lung is not None and lesion is not None:
            rows.append({"image": p, "stem": p.stem, "lung": lung, "lesion": lesion})
    return rows


# metrics
def topk_iou(scores, fg, pool):
    idx = np.where(pool)[0]
    if len(idx) == 0:
        return 0.0
    sub_fg = fg[idx]
    k = int(sub_fg.sum())
    if k == 0:
        return 0.0
    top = np.argsort(scores[idx])[::-1][:k]
    pred = np.zeros_like(sub_fg, dtype=bool)
    pred[top] = True
    inter = np.logical_and(pred, sub_fg).sum()
    union = np.logical_or(pred, sub_fg).sum()
    return float(inter / (union + 1e-8))

def random_topk_iou(fg, pool, n_trials=100):
    idx = np.where(pool)[0]
    if len(idx) == 0:
        return 0.0
    sub_fg = fg[idx]
    k = int(sub_fg.sum())
    if k == 0:
        return 0.0
    vals = []
    for _ in range(n_trials):
        chosen = np.random.permutation(len(idx))[:k]
        pred = np.zeros_like(sub_fg, dtype=bool)
        pred[chosen] = True
        inter = np.logical_and(pred, sub_fg).sum()
        union = np.logical_or(pred, sub_fg).sum()
        vals.append(float(inter / (union + 1e-8)))
    return float(np.mean(vals))


# masking
def cosine_torch(a, b):
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))

def occlude_patch_pixels(image, indices, value=0.0):
    pv = preprocess(image).clone()
    for idx in indices:
        r, c = int(idx) // NUM_PATCHES_1D, int(idx) % NUM_PATCHES_1D
        pv[0, :, r * PATCH_SIZE:(r + 1) * PATCH_SIZE, c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = value
    return pv

def masking_curve_occlusion(trunk, image, base_cls, scores, pool_idx, n_points=11, n_rand=3,
                            keep_idx=None):
    pool_idx = np.asarray(pool_idx, dtype=int)
    pool_scores = scores[pool_idx]
    high = pool_idx[np.argsort(pool_scores)[::-1]]
    low = pool_idx[np.argsort(pool_scores)]
    counts = [int(round(f * len(pool_idx))) for f in np.linspace(0, 1, n_points)]

    def sim_after_drop(indices):
        if len(indices) == 0:
            return 1.0
        pv = occlude_patch_pixels(image, indices)
        cls_new = (cls_from_pixel_values(trunk, pv) if keep_idx is None
                   else cls_lung_from_pixel_values(trunk, pv, keep_idx))
        return cosine_torch(base_cls, cls_new)

    high_sims, rand_sims, low_sims = [], [], []
    for c in counts:
        if c <= 0:
            high_sims.append(1.0); rand_sims.append(1.0); low_sims.append(1.0)
            continue
        c = min(c, len(pool_idx))
        high_sims.append(sim_after_drop(high[:c]))
        low_sims.append(sim_after_drop(low[:c]))
        rand_vals = [sim_after_drop(np.random.choice(pool_idx, c, replace=False))
                     for _ in range(n_rand)]
        rand_sims.append(float(np.mean(rand_vals)))
    return {
        "x": np.linspace(0, 1, n_points).tolist(),
        "high": high_sims,
        "random": rand_sims,
        "low": low_sims,
    }

def masking_auc_drop(curve):
    x = np.array(curve["x"], dtype=np.float32)
    high = np.array(curve["high"], dtype=np.float32)
    rand = np.array(curve["random"], dtype=np.float32)
    low = np.array(curve["low"], dtype=np.float32)
    return {
        "auc_high": float(np.trapz(1 - high, x)),
        "auc_random": float(np.trapz(1 - rand, x)),
        "auc_low": float(np.trapz(1 - low, x)),
        "gap_high_random": float(np.trapz(1 - high, x) - np.trapz(1 - rand, x)),
    }


# panels
def score_to_grid(scores):
    grid = np.asarray(scores, dtype=np.float32).reshape(NUM_PATCHES_1D, NUM_PATCHES_1D)
    valid = np.isfinite(grid) & (grid > -1e5)
    disp = np.full_like(grid, np.nan, dtype=np.float32)
    if valid.any():
        vals = grid[valid]
        disp[valid] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    return grid, disp

def save_input_panel(image, lung_cov, lesion_cov, pleural_cov, save_path, title):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    lung = lung_cov.reshape(NUM_PATCHES_1D, NUM_PATCHES_1D)
    lesion = lesion_cov.reshape(NUM_PATCHES_1D, NUM_PATCHES_1D)
    pleural = pleural_cov.reshape(NUM_PATCHES_1D, NUM_PATCHES_1D)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img, cmap="gray")
    ax.contour(np.array(Image.fromarray((lung > 0.5).astype(np.float32)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)),
               levels=[0.5], colors="lime", linewidths=1.2)
    ax.contour(np.array(Image.fromarray((pleural > 0.05).astype(np.float32)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)),
               levels=[0.5], colors="cyan", linewidths=1.0)
    ax.contour(np.array(Image.fromarray((lesion > 0.05).astype(np.float32)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)),
               levels=[0.5], colors="red", linewidths=1.2)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_heatmap_panel(image, scores, fg_cov, pool_cov, save_path, title):
    _, disp = score_to_grid(scores)
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    fg = fg_cov.reshape(NUM_PATCHES_1D, NUM_PATCHES_1D)
    pool = pool_cov.reshape(NUM_PATCHES_1D, NUM_PATCHES_1D)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img, cmap="gray")
    heat = Image.fromarray(np.nan_to_num(disp, nan=0.0)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    ax.imshow(heat, cmap="plasma", alpha=0.48)
    ax.contour(np.array(Image.fromarray((pool > 0.05).astype(np.float32)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)),
               levels=[0.5], colors="lime", linewidths=0.9)
    ax.contour(np.array(Image.fromarray((fg > 0.05).astype(np.float32)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)),
               levels=[0.5], colors="red", linewidths=1.1)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_pca_panel(image, patch_feats_full, fg_cov, pool_cov, save_path, title):
    feats = patch_feats_full.detach().cpu().numpy().astype(np.float32)
    valid = np.isfinite(feats).all(axis=1)
    if valid.sum() < 3:
        rgb = np.zeros((NUM_PATCHES_1D, NUM_PATCHES_1D, 3), dtype=np.float32)
    else:
        x = feats[valid]
        x = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        y = x @ vt[:3].T
        y = (y - y.min(axis=0, keepdims=True)) / (y.max(axis=0, keepdims=True) - y.min(axis=0, keepdims=True) + 1e-8)
        rgb_flat = np.zeros((NUM_PATCHES, 3), dtype=np.float32)
        rgb_flat[valid] = y[:, :3]
        rgb = rgb_flat.reshape(NUM_PATCHES_1D, NUM_PATCHES_1D, 3)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(Image.fromarray((rgb * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST))
    fg = fg_cov.reshape(NUM_PATCHES_1D, NUM_PATCHES_1D)
    pool = pool_cov.reshape(NUM_PATCHES_1D, NUM_PATCHES_1D)
    ax.contour(np.array(Image.fromarray((pool > 0.05).astype(np.float32)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)),
               levels=[0.5], colors="lime", linewidths=0.9)
    ax.contour(np.array(Image.fromarray((fg > 0.05).astype(np.float32)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)),
               levels=[0.5], colors="red", linewidths=1.1)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _kde_safe(vals, bw_method=0.15):
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size < 3 or np.ptp(vals) < 1e-8:
        return None
    try:
        return gaussian_kde(vals, bw_method=bw_method)
    except Exception:
        return None

def save_distribution_panel(scores, fg_labels, pool_labels, save_path, title=None, fg_name="Foreground"):
    valid = pool_labels & np.isfinite(scores) & (scores > -1e5)
    fg_vals = scores[valid & fg_labels]
    bg_vals = scores[valid & (~fg_labels)]

    all_vals = scores[valid]
    lo = float(min(all_vals.min(), -0.05)) if all_vals.size else -0.05
    hi = float(max(all_vals.max(), 1.0)) if all_vals.size else 1.0
    x = np.linspace(lo, hi, 300)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    kde_fg = _kde_safe(fg_vals)
    if kde_fg is not None:
        ax.fill_between(x, kde_fg(x), alpha=0.3, color="crimson")
        ax.plot(x, kde_fg(x), color="crimson", lw=2, label=f"{fg_name} μ={fg_vals.mean():.3f}")
        ax.axvline(fg_vals.mean(), color="crimson", ls="--", lw=1.5)
    kde_bg = _kde_safe(bg_vals)
    if kde_bg is not None:
        ax.fill_between(x, kde_bg(x), alpha=0.3, color="steelblue")
        ax.plot(x, kde_bg(x), color="steelblue", lw=2, label=f"In-pool BG μ={bg_vals.mean():.3f}")
        ax.axvline(bg_vals.mean(), color="steelblue", ls="--", lw=1.5)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Patch score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_masking_curve_panel(curve, save_path, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    ax.plot(curve["x"], curve["high"],   "-", lw=2, label="Drop high-score")
    ax.plot(curve["x"], curve["random"], "-", lw=2, label="Drop random")
    ax.plot(curve["x"], curve["low"],    "-", lw=2, label="Drop low-score")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Masked patch ratio")
    ax.set_ylabel("cos(CLS original, CLS masked)")
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_patchscore_heatmap_panel(scores, fg_cov, pool_cov, save_path, title=None,
                                  mark_topk=0, cmap="plasma"):
    _, disp = score_to_grid(scores)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(disp, cmap=cmap, vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# init
trunk, processor = load_model_and_processor()
_dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE, dtype=next(trunk.parameters()).dtype)
assert get_last_hidden(trunk, _dummy).shape[0] == 1 + NUM_PATCHES

rows = build_dataset_index(IMAGE_DIR, LUNG_DIR, LESION_DIR)


# scoring
def cosine_scores(patch_feats, query, eps=1e-8):
    pf = patch_feats.detach().float()
    pf = pf / (pf.norm(dim=-1, keepdim=True) + eps)
    q = query.detach().float()
    q = q / (q.norm() + eps)
    return (pf @ q).cpu().numpy()


def prior_cls(patch_feats, prior_cov, eps=1e-8):
    w = torch.as_tensor(prior_cov, dtype=torch.float32)
    return (w[:, None] * patch_feats.float()).sum(0) / (w.sum() + eps)


LUNG_THR = 0.50


def stage2_lung_only_scores(image, lung_cov, band_cov, lung_thr=LUNG_THR):
    lung_labels = lung_cov >= lung_thr
    keep_idx = np.where(lung_labels)[0]
    if len(keep_idx) < 4:
        return None

    cls_lung, patch_lung = extract_lung_only_features(trunk, image, keep_idx)
    q_band = prior_cls(patch_lung, band_cov[keep_idx])
    s_raw_k = cosine_scores(patch_lung, cls_lung)
    s_band_k = cosine_scores(patch_lung, q_band)

    s_raw = np.full(NUM_PATCHES, -1e6, dtype=np.float32);  s_raw[keep_idx] = s_raw_k
    s_band = np.full(NUM_PATCHES, -1e6, dtype=np.float32); s_band[keep_idx] = s_band_k

    dim = patch_lung.shape[-1]
    patch_full = torch.full((NUM_PATCHES, dim), float("nan"), dtype=patch_lung.dtype)
    patch_full[torch.as_tensor(keep_idx, dtype=torch.long)] = patch_lung

    return dict(lung_labels=lung_labels, keep_idx=keep_idx, cls_lung=cls_lung,
                patch_full=patch_full, s_raw=s_raw, s_band=s_band)


# metrics
def summarize(scores, fg, pool, cov):
    idx = np.where(pool)[0]
    order = idx[np.argsort(scores[idx])[::-1]]
    return dict(
        p1=bool(fg[order[:1]].any()),
        p3=bool(fg[order[:3]].any()),
        p5=bool(fg[order[:5]].any()),
        cov=float(cov[order[0]]),
        iou=topk_iou(scores, fg, pool),
        iou_rand=random_topk_iou(fg, pool),
    )


def masking_auc_whole(image, base_cls, scores):
    curve = masking_curve_occlusion(
        trunk, image, base_cls, scores, np.arange(NUM_PATCHES),
        n_points=MASKING_POINTS, n_rand=MASKING_RANDOM_TRIALS)
    return masking_auc_drop(curve)

def masking_auc_lung(image, base_cls, scores, keep_idx):
    curve = masking_curve_occlusion(
        trunk, image, base_cls, scores, keep_idx,
        n_points=MASKING_POINTS, n_rand=MASKING_RANDOM_TRIALS, keep_idx=keep_idx)
    return masking_auc_drop(curve)


# run
def select_rows(n_img=None, seed=0):
    if n_img is None or n_img <= 0 or n_img >= len(rows):
        return list(rows), "all"
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(rows), size=n_img, replace=False)
    return [rows[i] for i in sel], f"random-{n_img}(seed={seed})"


def _newblock(keys):
    return {v: {k: [] for k in ['p1', 'p3', 'p5', 'cov', 'iou', 'iou_rand']} for v in keys}

def _newmask(keys):
    return {v: {k: [] for k in ['auc_high', 'auc_random', 'auc_low', 'gap_high_random']} for v in keys}


def run(n_img=None, seed=0, band_px=18, masking=0, lung_thr=LUNG_THR, save=True):
    sel_rows, mode = select_rows(n_img, seed)
    ALL = np.ones(NUM_PATCHES, dtype=bool)

    S1 = _newblock(['raw', 'priorCLS'])
    S2 = _newblock(['raw', 'priorCLS'])
    rb1, rb2 = [], []
    MK = _newmask(['s1_raw', 's1_prior', 's2_raw', 's2_prior'])
    n_masked = 0

    for i, r in enumerate(sel_rows):
        img  = Image.open(r['image']).convert('RGB')
        lung = patch_coverage(r['lung'])
        les  = patch_coverage(r['lesion'])
        band = make_pleural_band_coverage(r['lung'], band_px=band_px)

        cls_w, pw = extract_whole_features(trunk, img)
        s_raw  = cosine_scores(pw, cls_w)
        s_lung = cosine_scores(pw, prior_cls(pw, lung))

        # stage1
        fg1 = lung > 0
        for nm, sc in [('raw', s_raw), ('priorCLS', s_lung)]:
            e = summarize(sc, fg1, ALL, lung)
            for k in e:
                S1[nm][k].append(e[k])
        rb1.append(float(fg1.mean()))

        # stage2
        st2 = stage2_lung_only_scores(img, lung, band, lung_thr=lung_thr)
        if st2 is not None:
            fg2 = les > 0
            lung_pool = st2['lung_labels']
            for nm, sc in [('raw', st2['s_raw']), ('priorCLS', st2['s_band'])]:
                e = summarize(sc, fg2, lung_pool, les)
                for k in e:
                    S2[nm][k].append(e[k])
            rb2.append(float(fg2[lung_pool].mean()) if lung_pool.any() else 0.0)

        # masking
        if masking and n_masked < masking and st2 is not None:
            for nm, sc in [('s1_raw', s_raw), ('s1_prior', s_lung)]:
                a = masking_auc_whole(img, cls_w, sc)
                for k in MK[nm]:
                    MK[nm][k].append(a[k])
            for nm, sc in [('s2_raw', st2['s_raw']), ('s2_prior', st2['s_band'])]:
                a = masking_auc_lung(img, st2['cls_lung'], sc, st2['keep_idx'])
                for k in MK[nm]:
                    MK[nm][k].append(a[k])
            n_masked += 1

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}  (masked {n_masked})")

    _report(S1, S2, rb1, rb2, MK, n_masked, mode)
    if save:
        _save(S1, S2, rb1, rb2, MK, n_masked, mode, band_px, lung_thr)
    return S1, S2, rb1, rb2, MK


# report
def _mean_block(block):
    return {v: {k: (float(np.mean(d[k])) if len(d[k]) else None) for k in d}
            for v, d in block.items()}

def _line(name, pool, d):
    print(f"{name:<18}{pool:<7}{np.mean(d['iou']):8.3f}{np.mean(d['p1']):7.3f}"
          f"{np.mean(d['p3']):7.3f}{np.mean(d['p5']):7.3f}")

def _header():
    print(f"{'method':<18}{'pool':<7}{'mIoU':>8}{'PiB@1':>7}{'PiB@3':>7}{'PiB@5':>7}")

def _report(S1, S2, rb1, rb2, MK, n_masked, mode):
    print(f"\n============  {mode}, threshold-free  ============")
    print("\n### STAGE 1 : whole image -> LUNG   (prior = lung coverage)")
    _header()
    _line('raw', 'whole', S1['raw'])
    _line('prior-CLS(lung)', 'whole', S1['priorCLS'])

    print("\n### STAGE 2 : LUNG-ONLY (pruned) -> PNEUMOTHORAX   (query steered by pleural band)")
    _header()
    _line('raw(lung-CLS)', 'lung', S2['raw'])
    _line('prior-CLS(band)', 'lung', S2['priorCLS'])

    if n_masked:
        print(f"\n### MASKING FAITHFULNESS  (on {n_masked} images; S1=all patches/whole, "
              f"S2=all lung tokens/lung-only; gap_h-r>0 => high-score patches drive the CLS)")
        print(f"{'curve':<12}{'auc_high':>9}{'auc_rand':>9}{'auc_low':>9}{'gap_h-r':>9}")
        for nm in ['s1_raw', 's1_prior', 's2_raw', 's2_prior']:
            d = MK[nm]
            if not len(d['auc_high']):
                continue
            print(f"{nm:<12}{np.mean(d['auc_high']):9.3f}{np.mean(d['auc_random']):9.3f}"
                  f"{np.mean(d['auc_low']):9.3f}{np.mean(d['gap_high_random']):9.3f}")

def _save(S1, S2, rb1, rb2, MK, n_masked, mode, band_px, lung_thr=LUNG_THR):
    out = {
        "config": {"model": MODEL_ID, "input_size": IMAGE_SIZE, "mode": mode, "band_px": band_px,
                   "lung_thr": lung_thr,
                   "stage2_mode": "lung_only_token_pruning + pleural_band_steered_query",
                   "masking_images": n_masked,
                   "masking_x_axis": "ratio_0_1",
                   "masking_stage1": "all_196_patches / whole-image re-encode",
                   "masking_stage2": "all_lung_tokens / lung-only re-encode",
                   "masking_points": MASKING_POINTS,
                   "masking_random_trials": MASKING_RANDOM_TRIALS},
        "stage1": _mean_block(S1),
        "stage2": _mean_block(S2),
        "masking": (_mean_block(MK) if n_masked else None),
        "random_baseline": {"stage1_lung_fraction": float(np.mean(rb1)),
                            "stage2_lesion_fraction_in_lung": float(np.mean(rb2))},
    }
    p = Path(OUT_BASE) / "prior_cls_two_stage_results.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved:", p)


# panels
PANEL_ROOT     = Path(OUT_BASE) / "panels_prior_cls"
PANEL_KINDS    = ["input", "pca", "heatmap", "patchscore_heatmap", "distribution", "masking_curve"]
PANEL_VARIANTS = ["raw", "priorCLS"]


def make_panel_dirs():
    for stage in ["stage1", "stage2"]:
        for variant in PANEL_VARIANTS:
            for k in PANEL_KINDS:
                (PANEL_ROOT / stage / variant / k).mkdir(parents=True, exist_ok=True)
    print("Panel root:", PANEL_ROOT)


def _render_variant_panels(stage, variant, img, stem, sc, patch_feats, fg_cov, pool_cov,
                           lung, les, band, fgname, base_cls, mask_pool_idx, keep_idx=None):
    root = PANEL_ROOT / stage / variant
    t = f"{stage} {variant}"
    save_input_panel(img, lung, les, band,
                     root / "input" / f"{stem}.png", t + ": input/prior")
    save_pca_panel(img, patch_feats, fg_cov, pool_cov,
                   root / "pca" / f"{stem}.png", t + ": PCA")
    save_heatmap_panel(img, sc, fg_cov, pool_cov,
                       root / "heatmap" / f"{stem}.png", t + ": heatmap")
    save_patchscore_heatmap_panel(sc, fg_cov, pool_cov,
                       root / "patchscore_heatmap" / f"{stem}.png", t + ": patch score")
    save_distribution_panel(sc, fg_cov > 0, pool_cov > 0,
                       root / "distribution" / f"{stem}.png", t + ": distribution",
                       fg_name=fgname)
    curve = masking_curve_occlusion(
        trunk, img, base_cls, sc, mask_pool_idx,
        n_points=MASKING_POINTS, n_rand=MASKING_RANDOM_TRIALS, keep_idx=keep_idx)
    save_masking_curve_panel(curve, root / "masking_curve" / f"{stem}.png",
                             t + ": masking curve")


def save_panels_for_image(row, band_px=18, lung_thr=LUNG_THR):
    stem = row["stem"]
    img  = Image.open(row["image"]).convert("RGB")
    lung = patch_coverage(row["lung"])
    les  = patch_coverage(row["lesion"])
    band = make_pleural_band_coverage(row["lung"], band_px=band_px)

    cls_w, pw = extract_whole_features(trunk, img)
    whole = np.ones(NUM_PATCHES, dtype=np.float32)
    all_idx = np.arange(NUM_PATCHES)

    # stage1
    for variant, sc in [("raw", cosine_scores(pw, cls_w)),
                        ("priorCLS", cosine_scores(pw, prior_cls(pw, lung)))]:
        _render_variant_panels("stage1", variant, img, stem, sc, pw,
                               fg_cov=lung, pool_cov=whole, lung=lung, les=les, band=band,
                               fgname="Lung", base_cls=cls_w, mask_pool_idx=all_idx, keep_idx=None)

    # stage2
    st2 = stage2_lung_only_scores(img, lung, band, lung_thr=lung_thr)
    if st2 is not None:
        keep_idx = st2["keep_idx"]
        cls_lung = st2["cls_lung"]
        patch_full = st2["patch_full"]
        lung_pool = st2["lung_labels"].astype(np.float32)
        for variant, sc in [("raw", st2["s_raw"]), ("priorCLS", st2["s_band"])]:
            _render_variant_panels("stage2", variant, img, stem, sc, patch_full,
                                   fg_cov=les, pool_cov=lung_pool, lung=lung, les=les, band=band,
                                   fgname="Pneumothorax", base_cls=cls_lung,
                                   mask_pool_idx=keep_idx, keep_idx=keep_idx)
    return stem


def save_panels_batch(n_img=None, seed=0, band_px=18, lung_thr=LUNG_THR):
    sel_rows, mode = select_rows(n_img, seed)
    make_panel_dirs()
    print(f"Rendering panels ({mode}) ...")
    failed = []
    for i, row in enumerate(sel_rows):
        try:
            save_panels_for_image(row, band_px=band_px, lung_thr=lung_thr)
        except Exception as e:
            failed.append({"image": str(row["image"]), "error": repr(e)})
            print("FAILED:", row["image"], repr(e))
        if (i + 1) % 20 == 0:
            print(f"  panels {i + 1}")
    print("Panels saved under:", PANEL_ROOT, "| failed:", len(failed))
    return failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=-1,
                    help="number of images; <=0 or omitted = ALL images (default)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (only used with --n / --panels)")
    ap.add_argument("--band_px", type=int, default=18, help="pleural band half-width in px")
    ap.add_argument("--lung_thr", type=float, default=LUNG_THR,
                    help="patch lung-coverage threshold to count as 'inside lung' (Stage 2 pruning)")
    ap.add_argument("--masking", type=int, default=0,
                    help="run masking faithfulness on the first N images (0 = off; expensive)")
    ap.add_argument("--panels", type=int, default=0,
                    help="render visualization panels for N images (0 = off, -1 = ALL). "
                         "When set, only panels are produced (skips the metric run).")
    ap.add_argument("--no-save", action="store_true", help="do not write results JSON")
    args = ap.parse_args()

    if args.panels != 0:
        save_panels_batch(n_img=args.panels, seed=args.seed, band_px=args.band_px,
                          lung_thr=args.lung_thr)
    else:
        run(n_img=args.n, seed=args.seed, band_px=args.band_px,
            masking=args.masking, lung_thr=args.lung_thr, save=not args.no_save)
