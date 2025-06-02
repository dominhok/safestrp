"""
Base classes and utilities for neural network heads.

This module provides foundational components for task-specific neural network heads
in the SafeStrp multi-task learning framework. It includes:

- BaseHead: Abstract base class for all task-specific heads
- PredictionHead: SSD-style prediction head for object detection
- PyramidPoolingModule: Multi-scale context aggregation for segmentation
- conv_unit: Basic convolution building block

All heads inherit from BaseHead and implement task-specific forward passes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod


def conv_unit(in_channels: int, 
              mid_channels: int, 
              out_channels: int, 
              stride_3x3: int = 1) -> nn.Sequential:
    """
    Create a basic convolution unit for feature extraction.
    
    This unit consists of:
    1. 1x1 convolution for channel reduction
    2. 3x3 convolution for feature extraction
    Both with batch normalization and ReLU activation.
    
    Args:
        in_channels: Number of input channels
        mid_channels: Number of intermediate channels (after 1x1 conv)
        out_channels: Number of output channels (after 3x3 conv)
        stride_3x3: Stride for the 3x3 convolution (default: 1)
    
    Returns:
        Sequential module containing the convolution unit
        
    Example:
        >>> conv_block = conv_unit(256, 128, 256, stride_3x3=2)
        >>> output = conv_block(input_tensor)  # (B, 256, H/2, W/2)
    """
    return nn.Sequential(
        # 1x1 convolution for efficient channel reduction
        nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        
        # 3x3 convolution for spatial feature extraction
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, 
                 padding=1, stride=stride_3x3, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class BaseHead(nn.Module, ABC):
    """
    Abstract base class for all task-specific heads.
    
    This class provides common functionality and interface for feature processing
    across different tasks (detection, segmentation, depth estimation).
    
    All concrete head implementations should inherit from this class and
    implement the forward() method.
    
    Attributes:
        Subclasses should define their specific attributes in __init__
        
    Methods:
        _init_weights(): Initialize weights with proper initialization
        get_feature_info(): Get information about expected feature channels
        forward(): Abstract method to be implemented by subclasses
    """
    
    def __init__(self) -> None:
        """Initialize the base head."""
        super().__init__()
    
    def _init_weights(self) -> None:
        """
        Initialize head weights with proper initialization schemes.
        
        Uses He initialization for Conv2d layers and standard initialization
        for BatchNorm2d layers. This helps with training stability and
        gradient flow in deep networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # He initialization for Conv2d (good for ReLU activation)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard initialization for BatchNorm
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def get_feature_info(self) -> Dict[str, int]:
        """
        Get information about expected backbone feature channels.
        
        Returns:
            Dictionary containing the expected number of channels for each
            backbone feature level (C3, C4, C5 from ResNet-50)
            
        Note:
            These values are specific to ResNet-50 backbone used in DSPNet
        """
        return {
            'backbone_c3_channels': 512,   # After layer2 (1/8 resolution)
            'backbone_c4_channels': 1024,  # After layer3 (1/16 resolution)
            'backbone_c5_channels': 2048   # After layer4 (1/32 resolution)
        }
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Abstract forward method to be implemented by subclasses.
        
        Args:
            *args: Variable positional arguments (typically feature tensors)
            **kwargs: Variable keyword arguments
            
        Returns:
            Task-specific output (detection boxes, segmentation masks, etc.)
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward() method")


class PredictionHead(nn.Module):
    """
    SSD-style prediction head for object detection.
    
    This head generates classification and regression predictions for each
    anchor location on a feature map. It's used in the multi-scale detection
    pipeline of DSPNet.
    
    The head predicts:
    - Classification scores for each class (including background)
    - Bounding box coordinate offsets relative to anchors
    
    Attributes:
        num_classes: Number of object classes (excluding background)
        num_anchors_per_location: Number of anchors per spatial location
        classifier_head: Conv2d layer for classification predictions
        regressor_head: Conv2d layer for regression predictions
    """
    
    def __init__(self, 
                 in_channels: int, 
                 num_anchors_per_location: int, 
                 num_classes: int) -> None:
        """
        Initialize the prediction head.
        
        Args:
            in_channels: Number of input feature channels
            num_anchors_per_location: Number of anchors per spatial location
            num_classes: Number of object classes (background is added automatically)
            
        Example:
            >>> head = PredictionHead(256, 4, 80)  # COCO dataset
            >>> cls_pred, reg_pred = head(feature_map)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location
        
        # Classification head: predicts class probabilities (including background)
        self.classifier_head = nn.Conv2d(
            in_channels,
            num_anchors_per_location * (num_classes + 1),  # +1 for background class
            kernel_size=3, 
            padding=1
        )
        
        # Regression head: predicts bbox coordinate offsets
        self.regressor_head = nn.Conv2d(
            in_channels,
            num_anchors_per_location * 4,  # 4 coordinates: [dx, dy, dw, dh]
            kernel_size=3, 
            padding=1
        )

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through prediction head.
        
        Args:
            feature_map: Input feature map of shape (B, C, H, W)
            
        Returns:
            Tuple containing:
            - cls_preds: Classification predictions (B, num_anchors * (num_classes + 1), H, W)
            - reg_preds: Regression predictions (B, num_anchors * 4, H, W)
            
        Note:
            Output tensors need to be reshaped for loss computation and NMS
        """
        cls_preds = self.classifier_head(feature_map)
        reg_preds = self.regressor_head(feature_map)
        
        return cls_preds, reg_preds


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module for multi-scale context aggregation.
    
    This module extracts features at multiple scales using adaptive pooling
    and combines them to capture both local and global context information.
    It's particularly effective for dense prediction tasks like segmentation.
    
    The module uses four pooling scales: 1x1, 2x2, 3x3, and 6x6, following
    the original PSPNet design.
    
    Attributes:
        pool_sizes: List of pooling scales [1, 2, 3, 6]
        pool_modules: ModuleList containing pooling branches
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the pyramid pooling module.
        
        Args:
            in_channels: Number of input feature channels
            out_channels: Total number of output channels (distributed across scales)
            
        Note:
            Each pooling branch outputs out_channels//4 features, so the total
            output will be exactly out_channels when concatenated.
            
        Example:
            >>> ppm = PyramidPoolingModule(2048, 512)
            >>> pooled_features = ppm(c5_features)  # (B, 512, H, W)
        """
        super().__init__()
        
        # Four pooling scales for multi-scale context
        self.pool_sizes = [1, 2, 3, 6]
        pool_out_channels = out_channels // 4  # Divide output channels equally
        
        self.pool_modules = nn.ModuleList([
            nn.Sequential(
                # Adaptive pooling to fixed size
                nn.AdaptiveAvgPool2d(pool_size),
                # 1x1 conv to reduce channels
                nn.Conv2d(in_channels, pool_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(pool_out_channels),
                nn.ReLU(inplace=True)
            )
            for pool_size in self.pool_sizes
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through pyramid pooling module.
        
        Args:
            x: Input feature tensor of shape (B, C, H, W)
            
        Returns:
            Pyramid pooled features of shape (B, out_channels, H, W)
            
        Process:
            1. Apply adaptive pooling at each scale
            2. Process through 1x1 conv + BN + ReLU
            3. Upsample back to original spatial size
            4. Concatenate all scales along channel dimension
        """
        input_size = x.size()[2:]  # Get spatial dimensions (H, W)
        pooled_features = []
        
        for pool_module in self.pool_modules:
            # Apply pooling and processing
            pooled = pool_module(x)
            
            # Upsample back to original size
            upsampled = F.interpolate(
                pooled, 
                size=input_size, 
                mode='bilinear', 
                align_corners=False
            )
            pooled_features.append(upsampled)
        
        # Concatenate all pooled features
        return torch.cat(pooled_features, dim=1) 