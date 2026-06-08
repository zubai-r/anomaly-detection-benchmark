# Training-Free Industrial Anomaly Detection: ResNet-50 vs DINOv2 vs DINOv3

A comparative benchmark of three frozen feature backbones — supervised **ResNet-50**,
self-supervised **DINOv2**, and self-supervised **DINOv3** — under a single, **training-free
PatchCore-style** anomaly detection pipeline, evaluated on all 15 categories of the
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

No backbone is fine-tuned. Each is used purely as a frozen patch-feature extractor; a
PatchCore memory bank (greedy coreset) is built from the *normal* training images and
nearest-neighbour distances at test time produce image- and pixel-level anomaly scores.

## Results (average over 15 MVTec AD categories)

| Backbone  | Image AUROC | Image F1-Max | Pixel AUROC | Pixel AUPRO | Latency |
| --------- | ----------- | ------------ | ----------- | ----------- | ------- |
| ResNet-50 | 91.74%      | 94.26%       | 95.45%      | 81.03%      | 1.79 ms |
| DINOv2    | **94.36%**  | **95.67%**   | **95.54%**  | **83.90%**  | 2.01 ms |
| DINOv3    | 92.45%      | 94.22%       | 94.69%      | 79.00%      | 2.10 ms |

Full per-category tables: [`results/collated_results.md`](results/collated_results.md).

## Repository contents

```
anomaly_detection/
├── backbones.py                    # frozen feature extractors (ResNet-50, DINOv2, DINOv3)
├── dataset.py                      # MVTec AD download + Dataset
├── patchcore.py                    # PatchCore memory bank + coreset + scoring
├── metrics.py                      # image/pixel AUROC, F1-Max, AUPRO
├── eval.py                         # single category × backbone evaluation
├── run_all_evals.py                # full 15×3 sweep + collated report
└── visualize.py                    # heatmaps, PCA overlays, t-SNE
results/
└── collated_results.md            # benchmark report (numbers above)
```

Only our pipeline is in this repo. The backbones themselves are **baselines we did not
author** — they are pulled from upstream (see links below), not vendored here.

## Baselines (external — not included in this repo)

- **DINOv3** — code: https://github.com/facebookresearch/dinov3 ·
  weights `dinov3_vits16_pretrain_lvd1689m`: https://github.com/facebookresearch/dinov3#pretrained-models
- **DINOv2** — https://github.com/facebookresearch/dinov2 (auto-loaded via `torch.hub`)
- **ResNet-50** — `IMAGENET1K_V2` weights from `torchvision.models` (auto-downloaded)
- **MVTec AD dataset** — https://www.mvtec.com/company/research/datasets/mvtec-ad
  (auto-downloaded at runtime from the HF mirror `hdtech/mvtech_anomaly_detection`, ~5.2 GB)

## Setup

> **Important:** `backbones.py` imports `dinov3.hub.backbones`, so this code must run with
> the **DINOv3 repository importable**. The simplest way is to place this `anomaly_detection/`
> package at the root of a cloned DINOv3 repo.

```bash
# 1. Clone the DINOv3 baseline repo
git clone https://github.com/facebookresearch/dinov3.git
cd dinov3

# 2. Add our pipeline at the repo root
git clone https://github.com/<your-username>/anomaly-detection-benchmark.git
cp -r anomaly-detection-benchmark/anomaly_detection ./anomaly_detection

# 3. Install dependencies
#    (install torch/torchvision for your CUDA first: https://pytorch.org)
pip install -r anomaly-detection-benchmark/requirements.txt

# 4. Download the DINOv3 ViT-S/16 weights to the repo root
#    file: dinov3_vits16_pretrain_lvd1689m-08c60483.pth
#    (from https://github.com/facebookresearch/dinov3#pretrained-models)
```

Resulting layout that the commands below expect:

```
dinov3/                      # cloned baseline repo (provides `dinov3` import)
├── anomaly_detection/       # our package
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
└── data/                    # MVTec AD (auto-created on first run)
```

## Running

All commands are run from the DINOv3 repo root.

```bash
# Single category, single backbone (MVTec AD auto-downloads on first run)
python -m anomaly_detection.eval --category bottle --backbone dinov3

# Other backbones / categories
python -m anomaly_detection.eval --category screw --backbone dinov2
python -m anomaly_detection.eval --category grid  --backbone resnet50

# Full benchmark: all 15 categories × 3 backbones, writes results/collated_results.md
python -m anomaly_detection.run_all_evals
```

Useful `eval.py` flags: `--coreset_ratio` (memory-bank size, default `0.01`),
`--layer_idx` (ViT layer to read, default `9`), `--input_size` (default `224`),
`--weights_path`, `--data_dir`, `--save_dir`.

Per-run outputs land in `results/`: `eval_<category>_<backbone>.txt` plus heatmap / PCA /
t-SNE PNGs. A GPU is recommended but the code falls back to CPU.

## Method summary

1. **Features** — read a single intermediate layer from each frozen backbone
   (ViT layer 9 for DINOv2/DINOv3; concatenated `layer2`+`layer3` for ResNet-50).
2. **Memory bank** — locally aggregate + L2-normalize patch features from normal training
   images, then greedy-coreset subsample (`coreset_ratio`).
3. **Scoring** — nearest-neighbour distance of each test patch to the bank gives the
   pixel anomaly map; the max patch score gives the image score.
4. **Metrics** — image AUROC / F1-Max, pixel AUROC, AUPRO, and per-image latency.

## License

This project is released under the [MIT License](LICENSE). Note that the external
baselines (DINOv3, DINOv2, ResNet-50 weights, MVTec AD) carry their own separate
licenses — see their upstream links above.
