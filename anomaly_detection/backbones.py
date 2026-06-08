import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# Import DINOv3 model from the local repository
from dinov3.hub.backbones import dinov3_vits16

class DINOv3Wrapper(nn.Module):
    def __init__(self, weights_path="dinov3_vits16_pretrain_lvd1689m-08c60483.pth", layer_idx=9):
        """
        Wrapper for local DINOv3 ViT-S/16 backbone.
        Args:
            weights_path (str): Local path to downloaded weights.
            layer_idx (int): 1-indexed layer to extract features from (e.g. 9/12).
        """
        super().__init__()
        print(f"Initializing DINOv3 (ViT-S/16) backbone from local weights '{weights_path}'...")
        # Load backbone natively using original codebase
        self.model = dinov3_vits16(pretrained=True, weights=weights_path)
        self.model.eval()
        
        # Internally, DinoVisionTransformer layers are 0-indexed.
        # Layer 9 (1-indexed) corresponds to index 8.
        self.layer_idx = layer_idx - 1 
        self.patch_size = 16
        
    def forward(self, x):
        """
        Extracts features from the specified intermediate layer.
        Returns:
            features (Tensor): Feature map of shape [B, C, H_patches, W_patches]
            grid_h (int): Height of patch grid.
            grid_w (int): Width of patch grid.
        """
        B, C, H, W = x.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        # Extract features using the repository's native get_intermediate_layers
        # reshape=True reshapes output to [B, embed_dim, grid_h, grid_w] and handles CLS/storage token stripping
        feats = self.model.get_intermediate_layers(x, n=[self.layer_idx], reshape=True, norm=True)
        out = feats[0]
        
        return out, grid_h, grid_w


class DINOv2Wrapper(nn.Module):
    def __init__(self, layer_idx=9):
        """
        Wrapper for DINOv2 ViT-S/14 backbone.
        Args:
            layer_idx (int): 1-indexed layer to extract features from (e.g. 9/12).
        """
        super().__init__()
        print("Initializing DINOv2 (ViT-S/14) backbone from PyTorch Hub...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval()
        
        self.layer_idx = layer_idx - 1
        self.patch_size = 14
        
    def forward(self, x):
        B, C, H, W = x.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        # DINOv2 Hub model also has get_intermediate_layers
        feats = self.model.get_intermediate_layers(x, n=[self.layer_idx], reshape=True, norm=True)
        out = feats[0]
        
        return out, grid_h, grid_w


class ResNet50Wrapper(nn.Module):
    def __init__(self):
        """
        Wrapper for supervised ResNet-50 backbone.
        Extracts layer2 and layer3 outputs and merges them to match PatchCore baseline.
        """
        super().__init__()
        print("Initializing ImageNet-supervised ResNet-50 backbone...")
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet.eval()
        
        # Use torchvision feature extractor to get intermediate layers
        self.feature_extractor = create_feature_extractor(
            resnet, 
            return_nodes={'layer2': 'layer2', 'layer3': 'layer3'}
        )
        
    def forward(self, x):
        """
        Extracts layer2 and layer3 maps, downsamples layer2, and concatenates them.
        Returns:
            features (Tensor): [B, 1536, 14, 14] at 224x224 input.
            grid_h (int): Height of feature map.
            grid_w (int): Width of feature map.
        """
        feats = self.feature_extractor(x)
        layer2_out = feats['layer2'] # Shape: [B, 512, 28, 28] at 224x224
        layer3_out = feats['layer3'] # Shape: [B, 1024, 14, 14] at 224x224
        
        # Downsample layer2_out to match layer3_out shape using average pooling
        layer2_pooled = F.avg_pool2d(layer2_out, kernel_size=2, stride=2)
        
        # Concatenate along channel dimension: 512 + 1024 = 1536 channels
        out = torch.cat([layer2_pooled, layer3_out], dim=1)
        
        grid_h, grid_w = out.shape[-2], out.shape[-1]
        return out, grid_h, grid_w


def get_backbone(name, device, layer_idx=9, weights_path="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
    """
    Factory function to retrieve backbones.
    """
    name_lower = name.lower()
    if "dinov3" in name_lower:
        wrapper = DINOv3Wrapper(weights_path=weights_path, layer_idx=layer_idx)
    elif "dinov2" in name_lower:
        wrapper = DINOv2Wrapper(layer_idx=layer_idx)
    elif "resnet50" in name_lower or "r50" in name_lower:
        wrapper = ResNet50Wrapper()
    else:
        raise ValueError(f"Unknown backbone name: {name}")
        
    wrapper = wrapper.to(device)
    wrapper.eval()
    
    # Freeze all parameters
    for p in wrapper.parameters():
        p.requires_grad = False
        
    return wrapper
