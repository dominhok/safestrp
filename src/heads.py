"""
Neural Network Heads for ThreeTaskDSPNet

통합된 Detection, Segmentation, Depth 헤드들:
- MultiTaskDetectionHead: SSD-style object detection (bbox only, distance removed)
- PyramidPoolingSegmentationHead: FCN-style semantic segmentation  
- DepthRegressionHead: Dense pixel-wise depth regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def conv_unit(in_channels: int, mid_channels: int, out_channels: int, stride_3x3: int = 1) -> nn.Sequential:
    """
    Basic convolution unit for extra layers.
    
    Args:
        in_channels: Input channels
        mid_channels: Intermediate channels  
        out_channels: Output channels
        stride_3x3: Stride for 3x3 convolution
    
    Returns:
        Sequential convolution unit
    """
    return nn.Sequential(
        # 1x1 convolution for channel reduction
        nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        
        # 3x3 convolution for feature extraction
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=stride_3x3, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class PredictionHead(nn.Module):
    """
    Prediction head for object detection and distance regression.
    
    Generates classification and regression predictions for each anchor location.
    The regression predictions include both bounding box coordinates and distance values.
    """
    
    def __init__(self, in_channels: int, num_anchors_per_location: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location
        
        # Classification head: predicts class probabilities (including background)
        self.classifier_head = nn.Conv2d(
            in_channels,
            num_anchors_per_location * (num_classes + 1),  # +1 for background
            kernel_size=3, padding=1
        )
        
        # Regression head: predicts bbox coordinates only (distance removed)
        self.regressor_head = nn.Conv2d(
            in_channels,
            num_anchors_per_location * 4,  # 4 for bbox (distance removed)
            kernel_size=3, padding=1
        )

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through prediction head.
        
        Args:
            feature_map: Input feature map of shape (B, C, H, W)
            
        Returns:
            tuple: (cls_preds, reg_preds)
                - cls_preds: (B, num_anchors, num_classes + 1)
                - reg_preds: (B, num_anchors, 4)  # 4 bbox
        """
        # Classification predictions
        cls_preds = self.classifier_head(feature_map)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.size(0), -1, self.num_classes + 1)
        
        # Regression predictions
        reg_preds = self.regressor_head(feature_map)
        reg_preds = reg_preds.permute(0, 2, 3, 1).contiguous()
        reg_preds = reg_preds.view(reg_preds.size(0), -1, 4)
        
        return cls_preds, reg_preds


class MultiTaskDetectionHead(nn.Module):
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
        Forward pass through detection head.
        
        Args:
            c3_feat: C3 features from backbone (B, 512, H/8, W/8)
            c4_feat: C4 features from backbone (B, 1024, H/16, W/16)  
            c5_feat: C5 features from backbone (B, 2048, H/32, W/32)
            
        Returns:
            tuple: (final_cls_preds, final_reg_preds)
                - final_cls_preds: (B, total_anchors, num_classes + 1)
                - final_reg_preds: (B, total_anchors, 4)  # 4 bbox
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

        # Apply prediction heads to each feature source
        all_cls_preds = []
        all_reg_preds = []

        for i, feature_map in enumerate(feature_sources):
            cls_preds, reg_preds = self.prediction_heads[i](feature_map)
            all_cls_preds.append(cls_preds)
            all_reg_preds.append(reg_preds)

        # Concatenate predictions from all feature sources
        final_cls_preds = torch.cat(all_cls_preds, dim=1)
        final_reg_preds = torch.cat(all_reg_preds, dim=1)

        return final_cls_preds, final_reg_preds

    def get_feature_info(self) -> dict:
        """Get information about feature levels and channels."""
        return {
            'num_levels': 7,
            'channels': self.feature_channels,
            'level_names': ['C3', 'C4', 'C5', 'EL1', 'EL2', 'EL3', 'EL4']
        }


class PyramidPoolingSegmentationHead(nn.Module):
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
        # PyramidPoolingModule 출력: 512 + 4*32 = 640 channels
        pyramid_out_channels = 512 + 128  # 640
        total_channels = 128 + 256 + 512 + pyramid_out_channels  # c3 + c4 + c5 + pyramid = 1536
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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
        pyramid_feat = self.pyramid_pooling(c5_processed)  # (B, 512, H/32, W/32)
        
        # Upsample all features to C3 size
        c4_upsampled = F.interpolate(c4_processed, size=target_size, mode='bilinear', align_corners=False)
        c5_upsampled = F.interpolate(c5_processed, size=target_size, mode='bilinear', align_corners=False)
        pyramid_upsampled = F.interpolate(pyramid_feat, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        fused_features = torch.cat([
            c3_processed,      # 128 channels
            c4_upsampled,      # 256 channels
            c5_upsampled,      # 512 channels
            pyramid_upsampled  # 640 channels (512 + 128)
        ], dim=1)  # Total: 1536 channels
        
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


class PyramidPoolingModule(nn.Module):
    """
    Pyramid pooling module for capturing multi-scale context.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Different pooling scales
        self.pool1 = nn.AdaptiveAvgPool2d(1)   # Global pooling
        self.pool2 = nn.AdaptiveAvgPool2d(2)   # 2x2 pooling
        self.pool3 = nn.AdaptiveAvgPool2d(3)   # 3x3 pooling
        self.pool6 = nn.AdaptiveAvgPool2d(6)   # 6x6 pooling
        
        # 1x1 convolutions to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv6 = nn.Conv2d(in_channels, out_channels // 4, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through pyramid pooling module.
        
        Args:
            x: Input feature map
            
        Returns:
            Pyramid pooled features
        """
        size = x.shape[2:]
        
        # Apply different pooling scales
        feat1 = F.interpolate(self.conv1(self.pool1(x)), size=size, mode='bilinear', align_corners=False)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), size=size, mode='bilinear', align_corners=False)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), size=size, mode='bilinear', align_corners=False)
        feat6 = F.interpolate(self.conv6(self.pool6(x)), size=size, mode='bilinear', align_corners=False)
        
        # Concatenate original features with pooled features
        return torch.cat([x, feat1, feat2, feat3, feat6], dim=1)


class DepthRegressionHead(nn.Module):
    """
    Dense depth regression head for pixel-wise depth estimation.
    
    This head takes high-level features and performs upsampling with skip connections
    to produce dense depth predictions at full resolution.
    """
    
    def __init__(self, input_channels: int = 2048, output_channels: int = 1):
        """
        Initialize depth regression head.
        
        Args:
            input_channels: Number of input channels from backbone (e.g., C5 features)
            output_channels: Number of output channels (typically 1 for depth)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Progressive upsampling with feature reduction
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final depth prediction layer
        self.depth_pred = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 1),
            nn.ReLU(inplace=True)  # Ensure positive depth values
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through depth regression head.
        
        Args:
            x: Input feature map from backbone (B, input_channels, H, W)
            
        Returns:
            depth_map: Dense depth predictions (B, output_channels, H_full, W_full)
        """
        # Progressive upsampling and feature refinement
        x = self.conv1(x)                                          # (B, 512, H, W)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 2x upsampling
        
        x = self.conv2(x)                                          # (B, 256, 2H, 2W)  
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 2x upsampling
        
        x = self.conv3(x)                                          # (B, 128, 4H, 4W)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 2x upsampling
        
        x = self.conv4(x)                                          # (B, 64, 8H, 8W)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)  # 4x upsampling
        
        # Final depth prediction
        depth_map = self.depth_pred(x)                             # (B, 1, 32H, 32W)
        
        return depth_map 