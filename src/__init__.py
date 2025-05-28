"""
ThreeTaskDSPNet Source Package

Detection + Surface + Depth 3태스크 멀티태스크 학습을 위한 핵심 모듈들.
"""

from .model import ThreeTaskDSPNet, create_model, load_pretrained_model
from .backbone import DSPNetBackbone
from .heads import MultiTaskDetectionHead, PyramidPoolingSegmentationHead, DepthRegressionHead
from .losses import SimpleTwoTaskLoss, FocalLoss, CrossEntropySegmentationLoss

__version__ = "1.0.0"
__author__ = "SafeStrp Team"

__all__ = [
    'ThreeTaskDSPNet',
    'create_model', 
    'load_pretrained_model',
    'DSPNetBackbone',
    'MultiTaskDetectionHead',
    'PyramidPoolingSegmentationHead',
    'DepthRegressionHead',
    'SimpleTwoTaskLoss',
    'FocalLoss',
    'CrossEntropySegmentationLoss'
] 