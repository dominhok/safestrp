"""
Detection head for object detection task.

Contains the MultiTaskDetectionHead for SSD-style object detection.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .base import BaseHead, PredictionHead, conv_unit


class MultiTaskDetectionHead(BaseHead):
    """
    Multi-task detection head for object detection.
    
    This head takes multi-scale features from the backbone and generates additional
    feature levels using extra convolution layers, then applies prediction heads
    to all feature levels for dense prediction.
    
    Based on the SSD architecture (distance estimation removed).
    """
    
    def __init__(self, 
                 num_classes: int, 
                 anchors_per_location_list: List[int], 
                 min_filter_extra_layers: int = 128):
        """
        Initialize the multi-task detection head.
        
        Args:
            num_classes: Number of object classes (excluding background)
            anchors_per_location_list: Number of anchors per location for each feature level
            min_filter_extra_layers: Minimum number of filters in extra layer conv units
        """
        super().__init__()
        
        # Store essential parameters
        self.num_classes = num_classes
        self.anchors_per_location_list = anchors_per_location_list
        
        if len(anchors_per_location_list) != 7:
            raise ValueError("anchors_per_location_list must have 7 elements for 7 feature sources.")

        # Input channels from backbone (ResNet-50 standard outputs)
        self.backbone_c3_channels = 512   # C3 output 
        self.backbone_c4_channels = 1024  # C4 output
        self.backbone_c5_channels = 2048  # C5 output

        # Extra layers configuration following DSPNet design
        # Creates 4 additional feature levels with decreasing spatial resolution
        
        # Extra Layer 1: C5 -> EL1
        el1_out_channels = 512
        el1_mid_channels = max(min_filter_extra_layers, el1_out_channels // 2)
        self.extra_layer1 = conv_unit(
            self.backbone_c5_channels, el1_mid_channels, el1_out_channels, stride_3x3=2
        )

        # Extra Layer 2: EL1 -> EL2  
        el2_out_channels = 256
        el2_mid_channels = max(min_filter_extra_layers, el2_out_channels // 2)
        self.extra_layer2 = conv_unit(
            el1_out_channels, el2_mid_channels, el2_out_channels, stride_3x3=2
        )

        # Extra Layer 3: EL2 -> EL3
        el3_out_channels = 256
        el3_mid_channels = max(min_filter_extra_layers, el3_out_channels // 2)
        self.extra_layer3 = conv_unit(
            el2_out_channels, el3_mid_channels, el3_out_channels, stride_3x3=2
        )

        # Extra Layer 4: EL3 -> EL4
        el4_out_channels = 128
        el4_mid_channels = max(min_filter_extra_layers, el4_out_channels // 2)
        self.extra_layer4 = conv_unit(
            el3_out_channels, el4_mid_channels, el4_out_channels, stride_3x3=2
        )

        # Store output channels for prediction heads
        self.feature_channels = [
            self.backbone_c3_channels,  # 512
            self.backbone_c4_channels,  # 1024
            self.backbone_c5_channels,  # 2048
            el1_out_channels,           # 512
            el2_out_channels,           # 256
            el3_out_channels,           # 256
            el4_out_channels            # 128
        ]

        # Prediction heads for each feature source
        self.prediction_heads = nn.ModuleList([
            PredictionHead(channels, anchors_per_location_list[i], num_classes)
            for i, channels in enumerate(self.feature_channels)
        ])

    def forward(self, c3_feat: torch.Tensor, c4_feat: torch.Tensor, c5_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-task detection head.
        
        Args:
            c3_feat: C3 feature maps (B, 512, H/8, W/8)
            c4_feat: C4 feature maps (B, 1024, H/16, W/16)
            c5_feat: C5 feature maps (B, 2048, H/32, W/32)
            
        Returns:
            tuple: (final_cls_preds, final_reg_preds)
                - final_cls_preds: (B, total_anchors, num_classes + 1)
                - final_reg_preds: (B, total_anchors, 4)
        """
        # Generate extra feature levels
        el1_feat = self.extra_layer1(c5_feat)      # (B, 512, H/64, W/64)
        el2_feat = self.extra_layer2(el1_feat)     # (B, 256, H/128, W/128)
        el3_feat = self.extra_layer3(el2_feat)     # (B, 256, H/256, W/256)
        el4_feat = self.extra_layer4(el3_feat)     # (B, 128, H/512, W/512)

        # Collect all feature sources
        feature_sources = [
            c3_feat,    # Source 0: C3
            c4_feat,    # Source 1: C4
            c5_feat,    # Source 2: C5
            el1_feat,   # Source 3: EL1
            el2_feat,   # Source 4: EL2
            el3_feat,   # Source 5: EL3
            el4_feat    # Source 6: EL4
        ]

        # Apply prediction heads to each feature source and flatten
        all_cls_preds = []
        all_reg_preds = []

        for i, feature_map in enumerate(feature_sources):
            # Get predictions: cls_preds (B, A*(C+1), H, W), reg_preds (B, A*4, H, W)
            cls_preds, reg_preds = self.prediction_heads[i](feature_map)
            
            # Reshape to SSD standard format: (B, A*(C+1), H, W) -> (B, H*W*A, C+1)
            B, _, H, W = cls_preds.shape
            A = self.anchors_per_location_list[i]  # anchors per location for this level
            C_plus_1 = self.num_classes + 1  # classes + background
            
            # Classification: (B, A*(C+1), H, W) -> (B, H*W*A, C+1)
            cls_preds = cls_preds.view(B, A, C_plus_1, H, W)  # (B, A, C+1, H, W)
            cls_preds = cls_preds.permute(0, 3, 4, 1, 2)  # (B, H, W, A, C+1)
            cls_preds = cls_preds.contiguous().view(B, H * W * A, C_plus_1)  # (B, H*W*A, C+1)
            
            # Regression: (B, A*4, H, W) -> (B, H*W*A, 4)
            reg_preds = reg_preds.view(B, A, 4, H, W)  # (B, A, 4, H, W)
            reg_preds = reg_preds.permute(0, 3, 4, 1, 2)  # (B, H, W, A, 4)
            reg_preds = reg_preds.contiguous().view(B, H * W * A, 4)  # (B, H*W*A, 4)
            
            all_cls_preds.append(cls_preds)
            all_reg_preds.append(reg_preds)

        # Concatenate predictions from all feature sources along anchor dimension
        final_cls_preds = torch.cat(all_cls_preds, dim=1)  # (B, total_anchors, C+1)
        final_reg_preds = torch.cat(all_reg_preds, dim=1)   # (B, total_anchors, 4)

        return final_cls_preds, final_reg_preds

    def get_feature_info(self) -> dict:
        """Get information about feature levels and channels."""
        return {
            'num_levels': 7,
            'channels': self.feature_channels,
            'level_names': ['C3', 'C4', 'C5', 'EL1', 'EL2', 'EL3', 'EL4']
        } 