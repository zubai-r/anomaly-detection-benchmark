import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from anomaly_detection.dataset import download_and_extract_dataset, verify_dataset_structure, MVTecDataset
from anomaly_detection.backbones import get_backbone
from anomaly_detection.patchcore import PatchCore
from anomaly_detection.metrics import compute_image_metrics, compute_pixel_auroc, compute_aupro
from anomaly_detection.visualize import create_pca_overlay, plot_tsne_scatter, save_comparison_grid

def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 Industrial Anomaly Detection Benchmarking")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category name")
    parser.add_argument("--backbone", type=str, default="dinov3", choices=["dinov3", "dinov2", "resnet50"], help="Backbone type")
    parser.add_argument("--layer_idx", type=int, default=9, help="1-indexed intermediate layer to read features from (for ViT)")
    parser.add_argument("--coreset_ratio", type=float, default=0.01, help="Percentage of features to keep in memory bank")
    parser.add_argument("--subsample_ratio", type=float, default=0.1, help="Subsample ratio before greedy coreset selection")
    parser.add_argument("--num_thresholds", type=int, default=100, help="Number of thresholds for AUPRO metric")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save visual results and logs")
    parser.add_argument("--weights_path", type=str, default="dinov3_vits16_pretrain_lvd1689m-08c60483.pth", help="Path to local DINOv3 weights")
    parser.add_argument("--input_size", type=int, default=224, help="Input image height/width")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    
    # 1. Download and extract MVTec AD dataset
    raw_path = download_and_extract_dataset(args.data_dir, category=args.category)
    dataset_path = verify_dataset_structure(raw_path)
    
    # 2. Setup DataLoaders with custom input size
    print(f"\nLoading '{args.category}' category with input size {args.input_size}...")
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor()
    ])
    
    train_dataset = MVTecDataset(root_path=dataset_path, category=args.category, split="train", transform=transform, mask_transform=mask_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    test_dataset = MVTecDataset(root_path=dataset_path, category=args.category, split="test", transform=transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # 3. Load Backbone
    backbone = get_backbone(args.backbone, device=device, layer_idx=args.layer_idx, weights_path=args.weights_path)
    
    # 4. Initialize PatchCore Model
    patchcore = PatchCore(coreset_ratio=args.coreset_ratio, pool_subsample_ratio=args.subsample_ratio)
    
    # 5. Build Memory Bank (Fit)
    t0 = time.time()
    patchcore.fit(backbone, train_loader, device=device)
    fit_time = time.time() - t0
    print(f"Memory bank construction completed in {fit_time:.2f} seconds.")
    
    # 6. Evaluation Loop
    print("\nRunning inference on test dataset...")
    test_labels = []
    test_masks = []
    pred_scores = []
    pred_maps = []
    
    # Collect a subset of patches for t-SNE visualization
    normal_patches_pool = []
    anomalous_patches_pool = []
    
    # Track latency
    total_time = 0.0
    num_images = 0
    
    # Track samples for qualitative output
    anomalous_samples = []
    
    with torch.no_grad():
        for images, masks, labels, paths in test_loader:
            images = images.to(device)
            
            # Predict anomaly scores and heatmaps
            t_start = time.time()
            batch_scores, batch_maps = patchcore.predict(backbone, images, device)
            total_time += (time.time() - t_start)
            num_images += images.shape[0]
            
            # Convert results to CPU
            pred_scores.extend(batch_scores.cpu().numpy())
            pred_maps.extend(batch_maps.cpu().numpy())
            test_labels.extend(labels.numpy())
            test_masks.extend(masks.cpu().numpy())
            
            # Extract features for t-SNE and PCA
            features, grid_h, grid_w = backbone(images)
            # Pool features locally: [B, grid_h * grid_w, C]
            pooled = patchcore._local_aggregation(features)
            # Normalize: [B, grid_h * grid_w, C]
            pooled_norm = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            
            # Collect patch-level ground truth mask mapping
            # Downsample ground truth masks to patch grid shape
            masks_resized = torch.nn.functional.interpolate(masks.to(device), size=(grid_h, grid_w), mode='area')
            masks_flat = masks_resized.view(images.shape[0], -1) # [B, grid_h * grid_w]
            
            for b in range(images.shape[0]):
                path = paths[b]
                label_val = labels[b].item()
                
                # Store one anomalous sample for heatmap visualization
                if label_val == 1 and len(anomalous_samples) < 3:
                    anomalous_samples.append({
                        "path": path,
                        "mask": masks[b],
                        "pred_map": batch_maps[b],
                        "patch_features": pooled_norm[b],
                        "grid_h": grid_h,
                        "grid_w": grid_w
                    })
                    
                # Collect features for t-SNE (limit count to keep it small)
                if len(normal_patches_pool) < 2000 or len(anomalous_patches_pool) < 2000:
                    norm_mask = masks_flat[b] <= 0.1
                    anom_mask = masks_flat[b] > 0.1
                    
                    if norm_mask.any():
                        normal_patches_pool.append(pooled_norm[b, norm_mask].cpu())
                    if anom_mask.any():
                        anomalous_patches_pool.append(pooled_norm[b, anom_mask].cpu())
                        
    # 7. Compute Metrics
    print("\nComputing metrics...")
    latency = (total_time / num_images) * 1000 # ms per image
    
    # Image-level metrics
    image_auroc, f1_max = compute_image_metrics(test_labels, pred_scores)
    
    # Pixel-level metrics
    pixel_auroc = compute_pixel_auroc(test_masks, pred_maps)
    
    # AUPRO
    aupro = compute_aupro(test_masks, pred_maps, num_thresholds=args.num_thresholds)
    
    print("\n========================================")
    print(f"BENCHMARK RESULTS FOR '{args.category}' using '{args.backbone}'")
    print("========================================")
    print(f"Image-level AUROC : {image_auroc * 100:.2f}%")
    print(f"Image-level F1-Max: {f1_max * 100:.2f}%")
    print(f"Pixel-level AUROC : {pixel_auroc * 100:.2f}%")
    print(f"Pixel-level AUPRO : {aupro * 100:.2f}%")
    print(f"Inference Latency : {latency:.2f} ms / image")
    print("========================================")
    
    # Write logs to file
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, f"eval_{args.category}_{args.backbone}.txt")
    with open(log_file, "w") as f:
        f.write(f"Category: {args.category}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Layer Index: {args.layer_idx}\n")
        f.write(f"Coreset Ratio: {args.coreset_ratio}\n")
        f.write(f"Image-level AUROC: {image_auroc * 100:.4f}%\n")
        f.write(f"Image-level F1-Max: {f1_max * 100:.4f}%\n")
        f.write(f"Pixel-level AUROC: {pixel_auroc * 100:.4f}%\n")
        f.write(f"Pixel-level AUPRO: {aupro * 100:.4f}%\n")
        f.write(f"Inference Latency: {latency:.4f} ms / image\n")
    print(f"Saved evaluation metrics to '{log_file}'.")
    
    # 8. Generate Visualizations
    print("\nGenerating visualizations...")
    
    # Save Heatmaps Comparison
    for idx, sample in enumerate(anomalous_samples):
        # Save a comparison map showing original, ground truth, and predicted heatmap
        map_path = os.path.join(args.save_dir, f"heatmap_{args.category}_{args.backbone}_{idx}.png")
        save_comparison_grid(sample["path"], sample["mask"], {args.backbone: sample["pred_map"]}, map_path)
        
        # Save PCA overlay map of patch features
        pca_map = create_pca_overlay(sample["patch_features"], sample["grid_h"], sample["grid_w"])
        pca_path = os.path.join(args.save_dir, f"pca_{args.category}_{args.backbone}_{idx}.png")
        plt.figure(figsize=(5, 5), dpi=300)
        plt.imshow(pca_map)
        plt.axis("off")
        plt.title(f"DINOv3 PCA Overlay ({args.category})")
        plt.savefig(pca_path, bbox_inches='tight')
        plt.close()
        print(f"Saved PCA map to '{pca_path}'.")
        
    # Save t-SNE Scatter Plot
    if len(normal_patches_pool) > 0 and len(anomalous_patches_pool) > 0:
        normal_patches = torch.cat(normal_patches_pool, dim=0)
        anomalous_patches = torch.cat(anomalous_patches_pool, dim=0)
        tsne_path = os.path.join(args.save_dir, f"tsne_{args.category}_{args.backbone}.png")
        plot_tsne_scatter(normal_patches, anomalous_patches, tsne_path)
        
if __name__ == "__main__":
    main()
