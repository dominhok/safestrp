"""
Task-specific heads for multi-task learning.

Contains detection, segmentation, depth, and cross-task heads.
"""

from .base import BaseHead, PyramidPoolingModule, PredictionHead, conv_unit
from .detection import MultiTaskDetectionHead
from .segmentation import PyramidPoolingSegmentationHead
from .depth import DepthRegressionHead
from .cross_task import CrossTaskProjectionHead, cosine_similarity_loss

__all__ = [
    # Base components
    'BaseHead',
    'PyramidPoolingModule', 
    'PredictionHead',
    'conv_unit',
    
    # Task-specific heads
    'MultiTaskDetectionHead',
    'PyramidPoolingSegmentationHead',
    'DepthRegressionHead',
    'CrossTaskProjectionHead',
    
    # Utility functions
    'cosine_similarity_loss'
] 