import os
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

def find_defect_sample(dataset_path, category):
    category_dir = os.path.join(dataset_path, category)
    if not os.path.exists(category_dir):
        category_dir = os.path.join(dataset_path, "mvtech_anomaly_detection", category)
        
    test_dir = os.path.join(category_dir, "test")
    defect_types = sorted([d for d in os.listdir(test_dir) if d != "good" and os.path.isdir(os.path.join(test_dir, d))])
    if not defect_types:
        raise ValueError(f"No defect types found in {test_dir}")
        
    defect_type = defect_types[0]
    defect_dir = os.path.join(test_dir, defect_type)
    images = sorted([img for img in os.listdir(defect_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        raise ValueError(f"No images found in {defect_dir}")
        
    return defect_type, images[0]

def generate_heatmap_grid(category, defect_type, img_name, dataset_path, device, save_path):
    train_dataset = MVTecDataset(root_path=dataset_path, category=category, split="train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    category_dir = os.path.join(dataset_path, category)
    if not os.path.exists(category_dir):
        category_dir = os.path.join(dataset_path, "mvtech_anomaly_detection", category)
        
    img_path = os.path.join(category_dir, "test", defect_type, img_name)
    mask_path = os.path.join(category_dir, "ground_truth", defect_type, f"{os.path.splitext(img_name)[0]}_mask.png")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(category_dir, "ground_truth", defect_type, f"{os.path.splitext(img_name)[0]}.png")
        
    prediction_maps = {}
    backbones = ["resnet50", "dinov2", "dinov3"]
    
    for bb_name in backbones:
        backbone = get_backbone(bb_name, device=device)
        patchcore = PatchCore(coreset_ratio=0.01, pool_subsample_ratio=0.1)
        patchcore.fit(backbone, train_loader, device=device)
        
        test_dataset = MVTecDataset(root_path=dataset_path, category=category, split="test")
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = test_dataset.transform(img_pil).unsqueeze(0).to(device)
        
        _, maps = patchcore.predict(backbone, img_tensor, device)
        prediction_maps[bb_name] = maps[0].cpu().numpy()
        
        del backbone, patchcore, img_tensor
        torch.cuda.empty_cache()
        
    num_cols = 5
    fig, axes = plt.subplots(1, num_cols, figsize=(15, 3), dpi=300)
    
    # Original Image
    img_pil_resized = Image.open(img_path).convert("RGB").resize((224, 224))
    axes[0].imshow(img_pil_resized)
    axes[0].axis("off")
    
    # Ground Truth Mask
    if os.path.exists(mask_path):
        gt_mask = Image.open(mask_path).convert("L").resize((224, 224))
        axes[1].imshow(gt_mask, cmap='gray')
    else:
        # fallback
        axes[1].imshow(np.zeros((224, 224)), cmap='gray')
    axes[1].axis("off")
    
    for idx, bb_name in enumerate(backbones):
        col_idx = 2 + idx
        heatmap = prediction_maps[bb_name][0]
        axes[col_idx].imshow(img_pil_resized)
        axes[col_idx].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[col_idx].axis("off")
        
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_tsne_plots(category, dataset_path, device, save_dir):
    train_dataset = MVTecDataset(root_path=dataset_path, category=category, split="train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    test_dataset = MVTecDataset(root_path=dataset_path, category=category, split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    backbones = ["dinov2", "dinov3"]
    
    for bb_name in backbones:
        backbone = get_backbone(bb_name, device=device)
        patchcore = PatchCore(coreset_ratio=0.01, pool_subsample_ratio=0.1)
        
        normal_patches_pool = []
        anomalous_patches_pool = []
        
        with torch.no_grad():
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
                            
        if len(normal_patches_pool) == 0 or len(anomalous_patches_pool) == 0:
            print(f"    Skipping t-SNE for {bb_name} on {category} - not enough patches.")
            continue
            
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
        ax.set_title(f"t-SNE of {bb_name.upper()} Features ({category})", fontsize=10, fontweight='bold')
        
        save_path = os.path.join(save_dir, f"tsne_{bb_name}_{category}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        del backbone, patchcore
        torch.cuda.empty_cache()

def generate_pca_map(category, defect_type, img_name, dataset_path, device, save_path):
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
        patchcore = PatchCore()
        pooled = patchcore._local_aggregation(features)
        pooled_norm = F.normalize(pooled, p=2, dim=-1)
        
    pca_img = create_pca_overlay(pooled_norm[0], grid_h, grid_w)
    
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(pca_img)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    del backbone
    torch.cuda.empty_cache()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    
    # 1. Verify and obtain dataset path
    raw_path = download_and_extract_dataset("data", category="carpet")
    dataset_path = verify_dataset_structure(raw_path)
    
    # Create target directory for all 15 categories presentation quality visuals
    save_dir = "presentation_visuals/all_categories"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nStarting automated visualization generation for all 15 categories...")
    
    for category in MVTEC_CATEGORIES:
        print(f"\n----------------------------------------")
        print(f"Processing category: '{category}'")
        print(f"----------------------------------------")
        
        try:
            # Discover sample image dynamically
            defect_type, img_name = find_defect_sample(dataset_path, category)
            print(f"Selected sample: test/{defect_type}/{img_name}")
            
            # A. Combined Heatmap Grid
            heatmap_path = os.path.join(save_dir, f"heatmap_{category}.png")
            generate_heatmap_grid(category, defect_type, img_name, dataset_path, device, heatmap_path)
            print(f"  Generated heatmap grid -> heatmap_{category}.png")
            
            # B. PCA Overlay map (DINOv3)
            pca_path = os.path.join(save_dir, f"pca_{category}.png")
            generate_pca_map(category, defect_type, img_name, dataset_path, device, pca_path)
            print(f"  Generated PCA map -> pca_{category}.png")
            
            # C. t-SNE scatter plots (DINOv2 and DINOv3)
            generate_tsne_plots(category, dataset_path, device, save_dir)
            print(f"  Generated t-SNE scatter plots -> tsne_dinov2/3_{category}.png")
            
        except Exception as e:
            print(f"Error processing category '{category}': {e}")
            
    print(f"\nAll visuals successfully generated and exported to the folder: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
