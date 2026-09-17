set -u
cd "$(dirname "$0")"
for d in models/*/; do
    name=$(basename "$d")
    echo "=============================================================="
    echo "### $name"
    echo "=============================================================="
    ( cd "$d" && python prior_cls_two_stage_eval.py "$@" ) \
        || echo "!! $name FAILED"
done
