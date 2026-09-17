# Beyond Registers: Diagnosing Lazy Aggregation in Vision Transformers for Pneumothorax Localization on Chest Radiographs

This repository contains the official code implementation for the paper **"Beyond Registers: Diagnosing Lazy Aggregation in Vision Transformers for Pneumothorax Localization on Chest Radiographs"**.

##Overview

While Vision Transformers (ViTs) achieve high performance in chest X-ray analysis, they often suffer from **"Lazy Aggregation"**—relying on non-pathological visual shortcuts rather than genuine lesions. To resolve this without retraining backbones, we propose a **training-free and parameter-free prior-guided attribution scheme** that corrects weak pathology alignment at the readout stage.

Instead of scoring patches against the default `[CLS]` token, our method constructs an anatomy-guided query using coverage-weighted pooling under two-stage spatial priors:
1. **Lung Prior**: Prunes non-lung tokens and restricts the global query strictly within the lung field.
2. **Pleural-Band Prior**: Directs attention to the peripheral zone where pneumothorax typically accumulates.
