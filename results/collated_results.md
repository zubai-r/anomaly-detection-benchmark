# MVTec AD Comparative Benchmark Report

This report summarizes the performance of three backbones (supervised ResNet-50, SSL DINOv2, and SSL DINOv3) under a training-free PatchCore-style pipeline.

## 1. Summary of Averages

| Backbone | Image-level AUROC | Image-level F1-Max | Pixel-level AUROC | Pixel-level AUPRO | Latency (ms/img) |
| --- | --- | --- | --- | --- | --- |
| resnet50 | 91.74% | 94.26% | 95.45% | 81.03% | 1.79 ms |
| dinov2 | 94.36% | 95.67% | 95.54% | 83.90% | 2.01 ms |
| dinov3 | 92.45% | 94.22% | 94.69% | 79.00% | 2.10 ms |

## 2. Per-Category Results

### Image-level AUROC

| Category | ResNet-50 | DINOv2 | DINOv3 |
| --- | --- | --- | --- |
| bottle | 100.00% | 100.00% | 100.00% |
| cable | 95.60% | 94.96% | 96.25% |
| capsule | 75.39% | 88.95% | 83.57% |
| carpet | 95.22% | 99.68% | 96.63% |
| grid | 68.42% | 98.66% | 99.00% |
| hazelnut | 99.96% | 99.86% | 97.82% |
| leather | 100.00% | 100.00% | 99.76% |
| metal_nut | 98.48% | 98.34% | 96.82% |
| pill | 91.19% | 96.15% | 89.12% |
| screw | 77.54% | 58.35% | 54.48% |
| tile | 99.93% | 100.00% | 99.96% |
| toothbrush | 90.83% | 93.06% | 94.44% |
| transistor | 95.33% | 94.71% | 93.17% |
| wood | 98.07% | 95.70% | 89.12% |
| zipper | 90.15% | 96.93% | 96.64% |

### Pixel-level AUROC

| Category | ResNet-50 | DINOv2 | DINOv3 |
| --- | --- | --- | --- |
| bottle | 97.98% | 98.28% | 98.05% |
| cable | 97.43% | 96.96% | 96.88% |
| capsule | 96.55% | 97.48% | 97.75% |
| carpet | 98.50% | 99.39% | 98.98% |
| grid | 88.62% | 98.38% | 96.21% |
| hazelnut | 98.50% | 98.94% | 98.47% |
| leather | 98.88% | 99.17% | 98.60% |
| metal_nut | 97.59% | 96.35% | 96.77% |
| pill | 97.59% | 96.85% | 96.19% |
| screw | 84.74% | 71.02% | 67.47% |
| tile | 95.62% | 97.85% | 97.49% |
| toothbrush | 96.81% | 96.93% | 96.53% |
| transistor | 95.75% | 97.76% | 96.30% |
| wood | 92.42% | 95.00% | 92.16% |
| zipper | 94.71% | 92.78% | 92.52% |

### Pixel-level AUPRO

| Category | ResNet-50 | DINOv2 | DINOv3 |
| --- | --- | --- | --- |
| bottle | 89.05% | 91.48% | 90.04% |
| cable | 86.59% | 89.39% | 86.67% |
| capsule | 76.81% | 78.86% | 80.41% |
| carpet | 90.20% | 94.76% | 89.72% |
| grid | 63.26% | 93.72% | 88.74% |
| hazelnut | 88.56% | 90.95% | 88.32% |
| leather | 95.10% | 95.95% | 91.02% |
| metal_nut | 85.02% | 89.95% | 85.01% |
| pill | 86.84% | 90.18% | 81.09% |
| screw | 57.63% | 33.61% | 34.70% |
| tile | 83.93% | 90.44% | 86.94% |
| toothbrush | 61.07% | 68.51% | 60.98% |
| transistor | 89.15% | 87.84% | 76.07% |
| wood | 81.25% | 82.22% | 66.60% |
| zipper | 80.94% | 80.67% | 78.62% |

### Inference Latency

| Category | ResNet-50 | DINOv2 | DINOv3 |
| --- | --- | --- | --- |
| bottle | 1.87 ms | 2.17 ms | 2.07 ms |
| cable | 1.63 ms | 1.78 ms | 1.85 ms |
| capsule | 1.60 ms | 2.04 ms | 1.99 ms |
| carpet | 1.55 ms | 1.96 ms | 2.00 ms |
| grid | 2.18 ms | 2.06 ms | 2.34 ms |
| hazelnut | 1.76 ms | 2.04 ms | 2.27 ms |
| leather | 1.65 ms | 1.81 ms | 1.93 ms |
| metal_nut | 1.78 ms | 2.09 ms | 2.34 ms |
| pill | 1.52 ms | 1.78 ms | 1.85 ms |
| screw | 1.42 ms | 1.78 ms | 1.81 ms |
| tile | 1.74 ms | 1.80 ms | 1.90 ms |
| toothbrush | 2.61 ms | 2.98 ms | 3.13 ms |
| transistor | 1.94 ms | 2.04 ms | 2.10 ms |
| wood | 1.97 ms | 2.03 ms | 2.13 ms |
| zipper | 1.60 ms | 1.78 ms | 1.84 ms |

