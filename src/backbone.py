"""
DSPNet Backbone Module

This module implements the backbone network for DSPNet using ResNet-50.
The backbone extracts multi-scale features (C3, C4, C5) that are used by 
detection and segmentation heads.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DSPNetBackbone(nn.Module):
    """
    DSPNet backbone based on ResNet-50.
    
    Extracts multi-scale features from input images:
    - C3: 1/8 resolution, 512 channels (from layer2)
    - C4: 1/16 resolution, 1024 channels (from layer3)  
    - C5: 1/32 resolution, 2048 channels (from layer4)
    
    Args:
        pretrained (bool): Whether to use ImageNet pretrained weights
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        
        # Initial stem layers (conv1 + bn1 + relu + maxpool)
        self.stem = nn.Sequential(
            resnet.conv1,    # 7x7, stride=2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool   # 3x3, stride=2
        )
        
        # ResNet stages
        self.layer1 = resnet.layer1  # C2: 1/4 resolution, 256 channels
        self.layer2 = resnet.layer2  # C3: 1/8 resolution, 512 channels
        self.layer3 = resnet.layer3  # C4: 1/16 resolution, 1024 channels
        self.layer4 = resnet.layer4  # C5: 1/32 resolution, 2048 channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through backbone.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            tuple: (c3_feat, c4_feat, c5_feat)
                - c3_feat: (B, 512, H/8, W/8)
                - c4_feat: (B, 1024, H/16, W/16)
                - c5_feat: (B, 2048, H/32, W/32)
        """
        # Stem processing: (B, 3, H, W) -> (B, 64, H/4, W/4)
        x = self.stem(x)
        
        # ResNet stages
        c2 = self.layer1(x)           # (B, 256, H/4, W/4)
        c3_feat = self.layer2(c2)     # (B, 512, H/8, W/8)
        c4_feat = self.layer3(c3_feat)  # (B, 1024, H/16, W/16)
        c5_feat = self.layer4(c4_feat)  # (B, 2048, H/32, W/32)

        return c3_feat, c4_feat, c5_feat

    def get_feature_channels(self) -> dict[str, int]:
        """Get output channel numbers for each feature level."""
        return {
            'c3': 512,
            'c4': 1024, 
            'c5': 2048
        } 