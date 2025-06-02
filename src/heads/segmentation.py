"""
Segmentation head for semantic segmentation task.

Contains PyramidPoolingSegmentationHead for dense pixel-wise prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead, PyramidPoolingModule


class PyramidPoolingSegmentationHead(BaseHead):
    """
    Pyramid pooling segmentation head for semantic segmentation.
    
    This head uses pyramid pooling to capture multi-scale context information
    and produces dense pixel-wise predictions. Based on PSPNet architecture.
    """
    
    def __init__(self, num_classes: int, c3_channels: int = 512, c4_channels: int = 1024, c5_channels: int = 2048):
        """
        Initialize pyramid pooling segmentation head.
        
        Args:
            num_classes: Number of segmentation classes
            c3_channels: Number of channels in C3 feature maps
            c4_channels: Number of channels in C4 feature maps  
            c5_channels: Number of channels in C5 feature maps
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Feature reduction layers for each scale
        self.c3_conv = nn.Sequential(
            nn.Conv2d(c3_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.c4_conv = nn.Sequential(
            nn.Conv2d(c4_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.c5_conv = nn.Sequential(
            nn.Conv2d(c5_channels, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Pyramid pooling module
        self.pyramid_pooling = PyramidPoolingModule(512, 128)
        
        # Final prediction layers
        # PyramidPoolingModule 출력: 128 channels (out_channels 그대로)
        pyramid_out_channels = 128
        # c5를 pyramid pooling으로 변환했으므로 c5_processed는 사용하지 않음
        total_channels = 128 + 256 + pyramid_out_channels  # c3 + c4 + pyramid = 512
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def forward(self, c3_feat: torch.Tensor, c4_feat: torch.Tensor, c5_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through segmentation head.
        
        Args:
            c3_feat: C3 features from backbone (B, 512, H/8, W/8)
            c4_feat: C4 features from backbone (B, 1024, H/16, W/16)
            c5_feat: C5 features from backbone (B, 2048, H/32, W/32)
            
        Returns:
            segmentation_logits: (B, num_classes, H, W)
        """
        # Get target size from C3 feature maps
        target_size = c3_feat.shape[2:]
        
        # Process features at each scale
        c3_processed = self.c3_conv(c3_feat)  # (B, 128, H/8, W/8)
        c4_processed = self.c4_conv(c4_feat)  # (B, 256, H/16, W/16)
        c5_processed = self.c5_conv(c5_feat)  # (B, 512, H/32, W/32)
        
        # Apply pyramid pooling to C5
        pyramid_feat = self.pyramid_pooling(c5_processed)  # (B, 640, H/32, W/32)
        
        # Upsample all features to C3 size
        c4_upsampled = F.interpolate(c4_processed, size=target_size, mode='bilinear', align_corners=False)
        c5_upsampled = F.interpolate(c5_processed, size=target_size, mode='bilinear', align_corners=False)
        pyramid_upsampled = F.interpolate(pyramid_feat, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        fused_features = torch.cat([
            c3_processed,      # 128 channels
            c4_upsampled,      # 256 channels
            pyramid_upsampled  # 128 channels
        ], dim=1)  # Total: 512 channels
        
        # Final prediction
        segmentation_logits = self.final_conv(fused_features)  # (B, num_classes, H/8, W/8)
        
        # Upsample to original resolution
        segmentation_logits = F.interpolate(
            segmentation_logits, 
            scale_factor=8, 
            mode='bilinear', 
            align_corners=False
        )  # (B, num_classes, H, W)
        
        return segmentation_logits 