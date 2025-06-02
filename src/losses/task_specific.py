"""
Task-specific loss functions.

Contains losses for segmentation and depth estimation tasks.
"""

import torch
import torch.nn as nn
from typing import Optional


class CrossEntropySegmentationLoss(nn.Module):
    """
    Cross-entropy loss for semantic segmentation with ignore index support.
    """
    
    def __init__(self,
                 ignore_index: int = 255,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Initialize segmentation loss.
        
        Args:
            ignore_index: Index to ignore in loss computation
            weight: Class weights for handling class imbalance
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=weight,
            reduction=reduction
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute segmentation loss.
        
        Args:
            predictions: Predicted segmentation logits (B, num_classes, H, W)
            targets: Ground truth segmentation masks (B, H, W)
            
        Returns:
            Segmentation loss value
        """
        return self.criterion(predictions, targets)


class DepthLoss(nn.Module):
    """
    Loss function for depth estimation.
    
    Supports various depth loss types with ignore value handling.
    """
    
    def __init__(self,
                 loss_type: str = 'l1',
                 reduction: str = 'mean',
                 ignore_value: float = 0.0):
        """
        Initialize depth loss.
        
        Args:
            loss_type: Type of loss ('l1', 'l2', 'smooth_l1')
            reduction: Loss reduction method
            ignore_value: Value to ignore in loss computation
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.reduction = reduction
        self.ignore_value = ignore_value
        
        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute depth loss.
        
        Args:
            predictions: Predicted depth values
            targets: Ground truth depth values
            
        Returns:
            Depth loss value
        """
        # Create mask to ignore certain values
        mask = (targets != self.ignore_value).float()
        
        # Compute loss
        loss = self.criterion(predictions, targets)
        loss = loss * mask
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 