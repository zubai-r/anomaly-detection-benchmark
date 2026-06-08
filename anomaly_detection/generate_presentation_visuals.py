import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch.utils.data import DataLoader

from anomaly_detection.dataset import MVTecDataset, download_and_extract_dataset, verify_dataset_structure
from anomaly_detection.backbones import get_backbone
from anomaly_detection.patchcore import PatchCore
from anomaly_detection.visualize import create_pca_overlay

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", 
    "hazelnut", "leather", "metal_nut", "pill", "screw", 
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

def load_aupro_results():
    results = {
        "resnet50": {},
        "dinov2": {},
        "dinov3": {}
    }
    
    # Parse individual text log files to collect exact AUPRO metrics
    for bb in results.keys():
        for cat in MVTEC_CATEGORIES:
            log_file = os.path.join("results", f"eval_{cat}_{bb}.txt")
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    for line in f:
                        if "Pixel-level AUPRO" in line:
                            val = line.split(":")[1].strip().replace("%", "")
                            results[bb][cat] = float(val)
            else:
                # Fallbacks in case individual logs were deleted or modified
                # Using the values from collated_results.md
                fallback_vals = {
                    "resnet50": {"bottle": 89.05, "cable": 86.59, "capsule": 76.81, "carpet": 90.20, "grid": 63.26, "hazelnut": 88.56, "leather": 95.10, "metal_nut": 85.02, "pill": 86.84, "screw": 57.63, "tile": 83.93, "toothbrush": 61.07, "transistor": 89.15, "wood": 81.25, "zipper": 80.94},
                    "dinov2": {"bottle": 91.48, "cable": 89.39, "capsule": 78.86, "carpet": 94.76, "grid": 93.72, "hazelnut": 90.95, "leather": 95.95, "metal_nut": 89.95, "pill": 90.18, "screw": 33.61, "tile": 90.44, "toothbrush": 68.51, "transistor": 87.84, "wood": 82.22, "zipper": 80.67},
                    "dinov3": {"bottle": 90.04, "cable": 86.67, "capsule": 80.41, "carpet": 89.72, "grid": 88.74, "hazelnut": 88.32, "leather": 91.02, "metal_nut": 85.01, "pill": 81.09, "screw": 34.70, "tile": 86.94, "toothbrush": 60.98, "transistor": 76.07, "wood": 66.60, "zipper": 78.62}
                }
                results[bb][cat] = fallback_vals[bb][cat]
                
    return results

def generate_bar_chart(results, save_path):
    print("Generating grouped bar chart...")
    categories = MVTEC_CATEGORIES
    x = np.arange(len(categories))
    width = 0.25
    
    r50_vals = [results["resnet50"][cat] for cat in categories]
    v2_vals = [results["dinov2"][cat] for cat in categories]
    v3_vals = [results["dinov3"][cat] for cat in categories]
    
    # Premium color palette (HSL-tailored feel)
    # ResNet-50 = Crimson/Soft Red, DINOv2 = Steel/Slate Blue, DINOv3 = Vibrant/Royal Blue
    colors = {
        "resnet50": "#E05A47",
        "dinov2": "#4A6FA5",
        "dinov3": "#1E3D59"
    }
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    rects1 = ax.bar(x - width, r50_vals, width, label='ResNet-50 (Supervised)', color=colors["resnet50"])
    rects2 = ax.bar(x, v2_vals, width, label='DINOv2 (SSL)', color=colors["dinov2"])
    rects3 = ax.bar(x + width, v3_vals, width, label='DINOv3 (SSL, Ours)', color=colors["dinov3"])
    
    ax.set_ylabel('Pixel AUPRO (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=11, fontweight='medium')
    ax.set_ylim(0, 105)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#DDDDDD')
    ax.set_axisbelow(True)
    
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Grouped bar chart saved to: {save_path}")

def generate_heatmap_grid(category, defect_type, img_name, dataset_path, device, save_path):
    print(f"Generating combined heatmap comparison grid for category '{category}'...")
    
    # Load dataset
    train_dataset = MVTecDataset(root_path=dataset_path, category=category, split="train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Find test image & mask
    category_dir = os.path.join(dataset_path, category)
    if not os.path.exists(category_dir):
        category_dir = os.path.join(dataset_path, "mvtech_anomaly_detection", category)
        
    img_path = os.path.join(category_dir, "test", defect_type, img_name)
    mask_path = os.path.join(category_dir, "ground_truth", defect_type, f"{os.path.splitext(img_name)[0]}_mask.png")
    
    # Compute prediction maps for each backbone
    prediction_maps = {}
    backbones = ["resnet50", "dinov2", "dinov3"]
    
    for bb_name in backbones:
        print(f"  Fitting and predicting with '{bb_name}'...")
        backbone = get_backbone(bb_name, device=device)
        patchcore = PatchCore(coreset_ratio=0.01, pool_subsample_ratio=0.1)
        patchcore.fit(backbone, train_loader, device=device)
        
        # Predict on single image
        test_dataset = MVTecDataset(root_path=dataset_path, category=category, split="test")
        # Load and transform image manually matching dataset transform
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = test_dataset.transform(img_pil).unsqueeze(0).to(device)
        
        _, maps = patchcore.predict(backbone, img_tensor, device)
        prediction_maps[bb_name] = maps[0].cpu().numpy()
        
        # Free memory
        del backbone, patchcore, img_tensor
        torch.cuda.empty_cache()
        
    # Plot combined grid
    num_cols = 5
    fig, axes = plt.subplots(1, num_cols, figsize=(15, 3), dpi=300)
    
    # Original Image
    img_pil_resized = Image.open(img_path).convert("RGB").resize((224, 224))
    axes[0].imshow(img_pil_resized)
    axes[0].axis("off")
    
    # Ground Truth Mask
    gt_mask = Image.open(mask_path).convert("L").resize((224, 224))
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].axis("off")
    
    # Heatmaps
    display_names = {
        "resnet50": "ResNet-50",
        "dinov2": "DINOv2",
        "dinov3": "DINOv3 (Ours)"
    }
    
    for idx, bb_name in enumerate(backbones):
        col_idx = 2 + idx
        heatmap = prediction_maps[bb_name][0] # remove channel dim
        
        # Overlay heatmap
        axes[col_idx].imshow(img_pil_resized)
        axes[col_idx].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[col_idx].axis("off")
        
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Heatmap comparison saved to: {save_path}")

def generate_tsne_plots(category, dataset_path, device, save_dir):
    print("Generating t-SNE scatter plots...")
    train_dataset = MVTecDataset(root_path=dataset_path, category=category, split="train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    test_dataset = MVTecDataset(root_path=dataset_path, category=category, split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    backbones = ["dinov2", "dinov3"]
    
    for bb_name in backbones:
        print(f"  Running feature extraction & t-SNE for '{bb_name}'...")
        backbone = get_backbone(bb_name, device=device)
        patchcore = PatchCore(coreset_ratio=0.01, pool_subsample_ratio=0.1)
        
        normal_patches_pool = []
        anomalous_patches_pool = []
        
        with torch.no_grad():
            # Collect test patches
            for images, masks, labels, _ in test_loader:
                images = images.to(device)
                features, grid_h, grid_w = backbone(images)
                pooled = patchcore._local_aggregation(features)
                pooled_norm = F.normalize(pooled, p=2, dim=-1)
                
                masks_resized = F.interpolate(masks.to(device), size=(grid_h, grid_w), mode='area')
                masks_flat = masks_resized.view(images.shape[0], -1)
                
                for b in range(images.shape[0]):
                    if len(normal_patches_pool) < 1000 or len(anomalous_patches_pool) < 1000:
                        norm_mask = masks_flat[b] <= 0.1
                        anom_mask = masks_flat[b] > 0.1
                        if norm_mask.any() and len(normal_patches_pool) < 1000:
                            normal_patches_pool.append(pooled_norm[b, norm_mask].cpu())
                        if anom_mask.any() and len(anomalous_patches_pool) < 1000:
                            anomalous_patches_pool.append(pooled_norm[b, anom_mask].cpu())
                            
        # Compute t-SNE (perplexity=30)
        normal_patches = torch.cat(normal_patches_pool, dim=0)
        anomalous_patches = torch.cat(anomalous_patches_pool, dim=0)
        
        n_norm = min(1000, normal_patches.shape[0])
        n_anom = min(1000, anomalous_patches.shape[0])
        
        normal_subset = normal_patches[torch.randperm(normal_patches.shape[0])[:n_norm]].numpy()
        anom_subset = anomalous_patches[torch.randperm(anomalous_patches.shape[0])[:n_anom]].numpy()
        
        data = np.concatenate([normal_subset, anom_subset], axis=0)
        labels = np.concatenate([np.zeros(n_norm), np.ones(n_anom)], axis=0)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = tsne.fit_transform(data)
        
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.scatter(
            embeddings[labels == 0, 0], embeddings[labels == 0, 1], 
            color='#4A6FA5', label='Normal Patches', alpha=0.6, edgecolors='w', s=25, linewidths=0.5
        )
        ax.scatter(
            embeddings[labels == 1, 0], embeddings[labels == 1, 1], 
            color='#E05A47', label='Defect Patches', alpha=0.7, edgecolors='w', s=25, linewidths=0.5
        )
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        ax.legend(frameon=True, facecolor='white', edgecolor='none')
        ax.set_title(f"t-SNE of {bb_name.upper()} Features", fontsize=12, fontweight='bold')
        
        save_path = os.path.join(save_dir, f"tsne_{bb_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        del backbone, patchcore
        torch.cuda.empty_cache()
        print(f"t-SNE scatter plot saved to: {save_path}")

def generate_pca_map(category, defect_type, img_name, dataset_path, device, save_path):
    print("Generating PCA overlay map...")
    
    # Setup backbone & load single test image
    backbone = get_backbone("dinov3", device=device)
    test_dataset = MVTecDataset(root_path=dataset_path, category=category, split="test")
    
    category_dir = os.path.join(dataset_path, category)
    if not os.path.exists(category_dir):
        category_dir = os.path.join(dataset_path, "mvtech_anomaly_detection", category)
        
    img_path = os.path.join(category_dir, "test", defect_type, img_name)
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = test_dataset.transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features, grid_h, grid_w = backbone(img_tensor)
        # Apply local spatial aggregation to features
        patchcore = PatchCore()
        pooled = patchcore._local_aggregation(features)
        pooled_norm = F.normalize(pooled, p=2, dim=-1)
        
    # Fit PCA and create overlay
    pca_img = create_pca_overlay(pooled_norm[0], grid_h, grid_w)
    
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(pca_img)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    del backbone
    torch.cuda.empty_cache()
    print(f"PCA map saved to: {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    
    # 1. Verify and obtain dataset path
    raw_path = download_and_extract_dataset("data", category="carpet")
    dataset_path = verify_dataset_structure(raw_path)
    
    # Create target directory for presentation quality visuals
    save_dir = "presentation_visuals"
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Bar Chart of AUPROs
    results = load_aupro_results()
    generate_bar_chart(results, os.path.join(save_dir, "barchart_aupro.png"))
    
    # 3. Heatmap Comparison Grid: Carpet (hole, image 000.png)
    generate_heatmap_grid("carpet", "hole", "000.png", dataset_path, device, os.path.join(save_dir, "heatmap_carpet.png"))
    
    # 4. Heatmap Comparison Grid: Grid (bent, image 000.png)
    generate_heatmap_grid("grid", "bent", "000.png", dataset_path, device, os.path.join(save_dir, "heatmap_grid.png"))
    
    # 5. t-SNE scatter plots (perplexity=30) for DINOv2 and DINOv3 on Carpet
    generate_tsne_plots("carpet", dataset_path, device, save_dir)
    
    # 6. PCA overlay maps for DINOv3 on Carpet (hole, image 000.png)
    generate_pca_map("carpet", "hole", "000.png", dataset_path, device, os.path.join(save_dir, "pca_carpet.png"))
    
    print(f"\nAll visuals successfully generated and exported to the folder: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
