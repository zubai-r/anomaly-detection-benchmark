import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchCore(nn.Module):
    def __init__(self, coreset_ratio=0.01, pool_subsample_ratio=0.1, k_nearest=5):
        """
        PatchCore Anomaly Detector.
        Args:
            coreset_ratio (float): Ratio of memory bank patches to keep in coreset (default 1%).
            pool_subsample_ratio (float): Ratio of initial patches to subsample before coreset selection (default 10%).
            k_nearest (int): Number of nearest neighbors to compute image-level reweighted score.
        """
        super().__init__()
        self.coreset_ratio = coreset_ratio
        self.pool_subsample_ratio = pool_subsample_ratio
        self.k_nearest = k_nearest
        self.register_buffer("memory_bank", torch.empty(0))
        
    def _local_aggregation(self, features):
        """
        Applies 3x3 local neighborhood average pooling on spatial feature maps.
        Args:
            features (Tensor): Feature map of shape [B, C, H, W]
        Returns:
            aggregated_features (Tensor): Aggregated patch vectors [B, H * W, C]
        """
        B, C, H, W = features.shape
        # Pad by 1 to maintain spatial dimensions
        pooled = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        # Reshape to [B, C, H * W] and permute to [B, H * W, C]
        pooled = pooled.flatten(2).permute(0, 2, 1)
        return pooled

    def fit(self, backbone, dataloader, device):
        """
        Builds the coreset memory bank using normal training images.
        """
        print("Extracting training patch features...")
        all_patches = []
        
        with torch.no_grad():
            for images, _, _, _ in dataloader:
                images = images.to(device)
                # Extract features: Shape [B, C, grid_h, grid_w]
                features, _, _ = backbone(images)
                
                # Perform local neighborhood aggregation: Shape [B, grid_h * grid_w, C]
                aggregated = self._local_aggregation(features)
                
                # L2 normalize all patch vectors to align backbone scales
                aggregated_norm = F.normalize(aggregated, p=2, dim=-1)
                
                # Flatten batch and spatial dimensions
                aggregated_flat = aggregated_norm.reshape(-1, aggregated_norm.shape[-1])
                all_patches.append(aggregated_flat.cpu())
                
        # Shape: [N_total_patches, C]
        patch_pool = torch.cat(all_patches, dim=0)
        print(f"Total training patch pool size: {patch_pool.shape[0]} vectors of dim {patch_pool.shape[1]}")
        
        # Coreset selection
        self.memory_bank = self._select_coreset(patch_pool, device)
        print(f"Memory bank built with coreset size: {self.memory_bank.shape[0]}")
        
    def _select_coreset(self, patch_pool, device):
        """
        Implements the k-center greedy algorithm with random pre-subsampling.
        """
        n_patches = patch_pool.shape[0]
        coreset_size = int(n_patches * self.coreset_ratio)
        coreset_size = max(1, coreset_size) # ensure at least 1 vector
        
        # Step 1: Pre-subsample patch pool randomly (standard PatchCore O(n^2) speedup)
        subsample_size = int(n_patches * self.pool_subsample_ratio)
        subsample_size = max(coreset_size, subsample_size) # must be >= coreset size
        
        print(f"Randomly pre-subsampling patch pool from {n_patches} to {subsample_size}...")
        indices = torch.randperm(n_patches)[:subsample_size]
        subsampled_pool = patch_pool[indices].to(device)
        
        # Step 2: K-center greedy selection
        print(f"Selecting {coreset_size} coreset patches using k-center greedy algorithm...")
        coreset_indices = []
        
        # Initialize centers: pick the first center (since features are L2 normalized,
        # norms are uniform, making this effectively an arbitrary/index-0 selection)
        norms = torch.norm(subsampled_pool, p=2, dim=1)
        first_center_idx = torch.argmax(norms).item()
        coreset_indices.append(first_center_idx)
        
        # Track minimum distance from each point in subsampled pool to any center selected so far
        # Initialize with distance to the first selected center
        first_center = subsampled_pool[first_center_idx].unsqueeze(0)
        # Using batched L2 distance
        min_distances = torch.cdist(subsampled_pool, first_center, p=2).squeeze(1)
        
        for step in range(1, coreset_size):
            if step % 100 == 0 or step == coreset_size - 1:
                print(f"  Selected {step}/{coreset_size} centers...")
                
            # Find the point that is furthest from its nearest center
            next_center_idx = torch.argmax(min_distances).item()
            coreset_indices.append(next_center_idx)
            
            # Calculate distance of all points to the new center
            new_center = subsampled_pool[next_center_idx].unsqueeze(0)
            distances_to_new = torch.cdist(subsampled_pool, new_center, p=2).squeeze(1)
            
            # Update minimum distances
            min_distances = torch.minimum(min_distances, distances_to_new)
            
        coreset_patches = subsampled_pool[coreset_indices].cpu()
        return coreset_patches

    def predict(self, backbone, test_images, device):
        """
        Computes image-level and pixel-level anomaly scores.
        Args:
            backbone: Feature extractor.
            test_images (Tensor): Batch of test images [B, 3, 224, 224]
            device: Torch device.
        Returns:
            image_scores (Tensor): [B] Anomaly score per image.
            anomaly_maps (Tensor): [B, 1, 224, 224] Anomaly heatmap per image.
        """
        self.memory_bank = self.memory_bank.to(device)
        
        # Extract features: Shape [B, C, grid_h, grid_w]
        features, grid_h, grid_w = backbone(test_images)
        B, C, _, _ = features.shape
        
        # Pool and normalize: Shape [B, grid_h * grid_w, C]
        aggregated = self._local_aggregation(features)
        aggregated_norm = F.normalize(aggregated, p=2, dim=-1)
        
        # Batch processing: compute nearest neighbor distances to prevent OOM
        # memory_bank shape: [coreset_size, C]
        image_scores = []
        anomaly_maps = []
        
        for b in range(B):
            # Shape: [num_patches, C]
            test_patches = aggregated_norm[b]
            
            # Compute pairwise L2 distances in batches
            # test_patches: [grid_h * grid_w, C], memory_bank: [coreset_size, C]
            # dists: [grid_h * grid_w, coreset_size]
            dists = torch.cdist(test_patches, self.memory_bank, p=2)
            
            # Find nearest neighbor distance for each patch
            # patch_scores: [grid_h * grid_w]
            patch_scores, nn_indices = dists.min(dim=1)
            
            # --- Pixel-level Anomaly Map ---
            # Reshape back to grid and bilinearly upsample to original resolution
            patch_grid = patch_scores.view(1, 1, grid_h, grid_w)
            anomaly_map = F.interpolate(patch_grid, size=(test_images.shape[-2], test_images.shape[-1]), mode='bilinear', align_corners=False)
            anomaly_maps.append(anomaly_map)
            
            # --- Image-level Anomaly Score ---
            # Find patch with maximum nearest neighbor distance (d_max)
            d_max_val, d_max_idx = torch.max(patch_scores, dim=0)
            d_max_val = d_max_val.item()
            d_max_idx = d_max_idx.item()
            
            # Top-K closest distances for the maximum patch
            # dists[d_max_idx] has shape [coreset_size]
            topk_dists, _ = torch.topk(dists[d_max_idx], k=self.k_nearest, largest=False)
            
            # Softmax reweighting heuristic: w = 1 - exp(d_max) / sum(exp(d_i))
            # calculated over the test patch's nearest neighbors (a robust heuristic
            # related to but distinct from the exact paper Eq. 7 PatchCore formula).
            # Subtracting max for numerical stability in exp
            stable_topk = topk_dists - d_max_val
            exp_dists = torch.exp(stable_topk)
            w = 1.0 - (exp_dists[0] / torch.sum(exp_dists))
            
            s_image = w.item() * d_max_val
            image_scores.append(s_image)
            
        # Convert lists to tensors
        image_scores = torch.tensor(image_scores, dtype=torch.float32)
        anomaly_maps = torch.cat(anomaly_maps, dim=0)
        
        return image_scores, anomaly_maps
