"""
Loss functions for SafeStrp multi-task learning.

Contains base losses, task-specific losses, utilities, and multi-task losses.
"""

# Base loss functions
from .base import FocalLoss, FocalSmoothL1Loss

# Task-specific losses
from .task_specific import CrossEntropySegmentationLoss, DepthLoss

# Utility functions
from .utils import iou_match_anchors, box_iou, hard_negative_mining

# Multi-task losses
from .multitask import SimpleTwoTaskLoss, UberNetMTPSLLoss

# Backward compatibility: import from original losses.py if needed
try:
    import sys
    import os
    import importlib.util
    
    # Try to import remaining classes from original losses.py
    losses_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'losses.py')
    if os.path.exists(losses_path):
        spec = importlib.util.spec_from_file_location("original_losses", losses_path)
        original_losses = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original_losses)
        
        # Import any missing classes
        if hasattr(original_losses, 'AdvancedTwoTaskLoss'):
            AdvancedTwoTaskLoss = original_losses.AdvancedTwoTaskLoss
        else:
            # Create dummy class
            class AdvancedTwoTaskLoss: pass
            
except Exception:
    # Create dummy class if import fails
    class AdvancedTwoTaskLoss: pass

__all__ = [
    # Base losses
    'FocalLoss',
    'FocalSmoothL1Loss',
    
    # Task-specific losses
    'CrossEntropySegmentationLoss',
    'DepthLoss',
    
    # Utility functions
    'iou_match_anchors',
    'box_iou',
    'hard_negative_mining',
    
    # Multi-task losses
    'SimpleTwoTaskLoss',
    'UberNetMTPSLLoss',
    'AdvancedTwoTaskLoss'  # Backward compatibility
] 