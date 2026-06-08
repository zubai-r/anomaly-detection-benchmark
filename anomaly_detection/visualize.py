import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
import os

def create_pca_overlay(patch_features, grid_h, grid_w):
    """
    Computes PCA on patch features and maps the top 3 components to RGB channels.
    Args:
        patch_features (Tensor): Patch features of shape [num_patches, C]
        grid_h (int): Height of patch grid.
        grid_w (int): Width of patch grid.
    Returns:
        pca_image (ndarray): [224, 224, 3] image with PCA values mapped to RGB.
    """
    # Detach and move to CPU
    x = patch_features.detach().cpu().numpy()
    
    # Fit PCA with 3 components
    pca = PCA(n_components=3, whiten=True)
    projected = pca.fit_transform(x)
    
    # Reshape back to grid of patches [grid_h, grid_w, 3]
    projected_grid = torch.from_numpy(projected).view(grid_h, grid_w, 3)
    
    # Sigmoid scaling to bound values into RGB [0, 1] range
    projected_colors = torch.nn.functional.sigmoid(projected_grid.mul(2.0))
    
    # Upsample to 224x224 resolution for cleaner visual overlays
    projected_colors = projected_colors.permute(2, 0, 1).unsqueeze(0) # [1, 3, grid_h, grid_w]
    upsampled = F.interpolate(projected_colors, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Convert to numpy array [224, 224, 3]
    pca_image = upsampled.squeeze(0).permute(1, 2, 0).numpy()
    return pca_image

def plot_tsne_scatter(normal_patches, anomalous_patches, save_path):
    """
    Computes t-SNE on patch features and plots a scatter plot showing normal vs anomalous patches.
    Args:
        normal_patches (Tensor): Normal patches from normal images [N_normal, C]
        anomalous_patches (Tensor): Defect patches from anomalous regions [N_anom, C]
        save_path (str): File path to save the generated scatter plot.
    """
    print(f"Running t-SNE on {normal_patches.shape[0]} normal and {anomalous_patches.shape[0]} anomalous patches...")
    
    # Limit number of patches to prevent excessive compute times (t-SNE is O(N^2))
    max_patches = 1000
    n_norm = min(max_patches, normal_patches.shape[0])
    n_anom = min(max_patches, anomalous_patches.shape[0])
    
    normal_subset = normal_patches[torch.randperm(normal_patches.shape[0])[:n_norm]].cpu().numpy()
    anom_subset = anomalous_patches[torch.randperm(anomalous_patches.shape[0])[:n_anom]].cpu().numpy()
    
    # Combine data and create labels
    data = np.concatenate([normal_subset, anom_subset], axis=0)
    labels = np.concatenate([np.zeros(n_norm), np.ones(n_anom)], axis=0)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(data)
    
    plt.figure(figsize=(8, 6), dpi=300)
    # Plot normal patches
    plt.scatter(
        embeddings[labels == 0, 0], embeddings[labels == 0, 1], 
        color='blue', label='Normal Patches', alpha=0.6, edgecolors='w', s=20
    )
    # Plot anomalous patches
    plt.scatter(
        embeddings[labels == 1, 0], embeddings[labels == 1, 1], 
        color='red', label='Defect Patches', alpha=0.6, edgecolors='w', s=20
    )
    
    plt.title("t-SNE of Patch Features (DINOv3 Space)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"t-SNE scatter plot saved successfully to {save_path}")

def save_comparison_grid(img_path, gt_mask, predicted_maps_dict, save_path):
    """
    Saves a comparison grid of Input, Ground Truth mask, and heatmaps for different backbones.
    Args:
        img_path (str): Path to original test image.
        gt_mask (Tensor): Ground-truth mask [1, 224, 224]
        predicted_maps_dict (dict): Dictionary mapping backbone name -> anomaly map [1, 224, 224]
        save_path (str): Output file path.
    """
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    gt = gt_mask.squeeze().cpu().numpy()
    
    num_cols = 2 + len(predicted_maps_dict)
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4), dpi=300)
    
    # 1. Original Image
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    # 2. Ground Truth Mask
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    
    # 3. Anomaly Heatmaps for each backbone
    for idx, (backbone_name, map_tensor) in enumerate(predicted_maps_dict.items()):
        col_idx = 2 + idx
        heatmap = map_tensor.squeeze().cpu().numpy()
        
        # Display heatmap overlaid on original grayscale image
        axes[col_idx].imshow(img)
        # Overlay heatmap using alpha transparency
        axes[col_idx].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[col_idx].set_title(f"{backbone_name} Heatmap")
        axes[col_idx].axis("off")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
