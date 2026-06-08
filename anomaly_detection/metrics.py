import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from scipy.ndimage import label
from scipy.integrate import trapezoid

def compute_image_metrics(y_true, y_scores):
    """
    Computes image-level metrics: AUROC and F1-max.
    Args:
        y_true (ndarray): Binary labels [N] (0 for normal, 1 for anomalous).
        y_scores (ndarray): Anomaly scores [N].
    Returns:
        auroc (float): Area Under the ROC curve.
        f1_max (float): Maximum F1-score across all thresholds.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    auroc = roc_auc_score(y_true, y_scores)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_max = np.max(f1_scores)
    
    return auroc, f1_max

def compute_pixel_auroc(y_true_masks, y_score_maps):
    """
    Computes pixel-level AUROC over the entire test set.
    Args:
        y_true_masks (ndarray): Binary ground-truth masks [N, H, W] or [N, 1, H, W].
        y_score_maps (ndarray): Predicted anomaly maps [N, H, W] or [N, 1, H, W].
    """
    # Flatten all pixels across the dataset
    y_true = np.array(y_true_masks).ravel()
    y_scores = np.array(y_score_maps).ravel()
    
    # Convert to binary int
    y_true = (y_true > 0.5).astype(np.int32)
    
    return roc_auc_score(y_true, y_scores)

def compute_aupro(y_true_masks, y_score_maps, num_thresholds=100, max_fpr=0.3):
    """
    Computes the Area Under Per-Region Overlap (AUPRO) metric.
    AUPRO integrates the Average PRO vs FPR curve up to max_fpr (usually 0.3)
    and normalizes by dividing by max_fpr.
    
    Args:
        y_true_masks (ndarray): Ground-truth masks of shape [N, 224, 224]
        y_score_maps (ndarray): Predicted heatmaps of shape [N, 224, 224]
        num_thresholds (int): Number of thresholds to evaluate.
        max_fpr (float): FPR integration threshold (default 0.3).
    """
    print("Computing AUPRO (connected component labeled)...")
    y_true_masks = np.array(y_true_masks).squeeze() # Ensure shape [N, H, W]
    y_score_maps = np.array(y_score_maps).squeeze() # Ensure shape [N, H, W]
    
    # 1. Connected components labeling of ground-truth masks
    # We find all contiguous defect regions across the dataset
    regions = []
    total_normal_pixels = 0
    total_anomalous_pixels = 0
    
    # Normal pixel masks (where gt mask is 0)
    normal_pixels_masks = []
    
    for i in range(len(y_true_masks)):
        mask = y_true_masks[i]
        scores = y_score_maps[i]
        
        # Accumulate normal pixels
        normal_mask = (mask == 0)
        normal_pixels_masks.append(scores[normal_mask])
        total_normal_pixels += np.sum(normal_mask)
        
        # Find connected defect components
        labeled, num_features = label(mask)
        for r_id in range(1, num_features + 1):
            region_mask = (labeled == r_id)
            total_anomalous_pixels += np.sum(region_mask)
            
            # Store the scores inside this specific defect region
            regions.append({
                "scores": scores[region_mask],
                "size": np.sum(region_mask)
            })
            
    # If there are no anomalous regions (e.g. only normal images are tested), AUPRO is not defined
    if len(regions) == 0:
        return 0.0
        
    # All normal scores concatenated
    normal_scores = np.concatenate(normal_pixels_masks)
    
    # Determine thresholds spanning from min to max predicted score
    min_score = np.min(y_score_maps)
    max_score = np.max(y_score_maps)
    thresholds = np.linspace(min_score, max_score, num_thresholds)
    
    # Arrays to store FPR and Average PRO at each threshold
    fpr_list = []
    pro_list = []
    
    for t in thresholds:
        # False Positive Rate: normal pixels predicted as anomalous / total normal pixels
        fps = np.sum(normal_scores > t)
        fpr = fps / total_normal_pixels
        
        # Per-Region Overlap: average proportion of overlap for each defect region
        pro_overlaps = []
        for r in regions:
            overlap = np.sum(r["scores"] > t) / r["size"]
            pro_overlaps.append(overlap)
            
        avg_pro = np.mean(pro_overlaps)
        
        fpr_list.append(fpr)
        pro_list.append(avg_pro)
        
    fpr_list = np.array(fpr_list)
    pro_list = np.array(pro_list)
    
    # Sort by FPR (fpr should go from 0 to 1)
    sort_idx = np.argsort(fpr_list)
    fpr_list = fpr_list[sort_idx]
    pro_list = pro_list[sort_idx]
    
    # 2. Integrate curve up to max_fpr (0.3)
    # Filter points where FPR <= max_fpr
    valid_idx = fpr_list <= max_fpr
    fpr_sub = fpr_list[valid_idx]
    pro_sub = pro_list[valid_idx]
    
    # If the last FPR value in our subset is less than max_fpr, we interpolate the PRO at exactly max_fpr
    if len(fpr_sub) > 0 and fpr_sub[-1] < max_fpr:
        # Find the first index where FPR > max_fpr
        first_greater_idx = np.where(fpr_list > max_fpr)[0]
        if len(first_greater_idx) > 0:
            idx = first_greater_idx[0]
            # Interpolate PRO linearly
            fpr_prev = fpr_list[idx - 1]
            fpr_next = fpr_list[idx]
            pro_prev = pro_list[idx - 1]
            pro_next = pro_list[idx]
            
            # Linear interpolation
            pro_interp = pro_prev + (pro_next - pro_prev) * (max_fpr - fpr_prev) / (fpr_next - fpr_prev)
            
            fpr_sub = np.append(fpr_sub, max_fpr)
            pro_sub = np.append(pro_sub, pro_interp)
            
    # Integrate using the trapezoidal rule
    if len(fpr_sub) < 2:
        aupro = 0.0
    else:
        # trapezoid integrates y (pro_sub) along x (fpr_sub)
        aupro = trapezoid(pro_sub, fpr_sub)
        
    # 3. Normalize: divide by max_fpr (0.3) to scale the metric to [0, 1]
    # (Without division, the max possible area would be 0.3, making scores 3.33x too small)
    normalized_aupro = aupro / max_fpr
    
    return normalized_aupro
