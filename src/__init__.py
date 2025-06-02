"""
SafeStrp - Sidewalk Safety for All

멀티태스크 딥러닝 모델을 사용한 안전한 보행 환경 분석 시스템.
UberNet + MTPSL 하이브리드 구조로 Detection, Surface Segmentation, Depth Estimation을 수행합니다.

주요 모듈:
- core: 핵심 모델 구성요소 (모델, 백본, 앵커)
- heads: 태스크별 헤드들 (detection, segmentation, depth, cross-task)
- losses: 손실 함수들 (base, task-specific, multitask, utils)
- data: 데이터 로딩 및 처리
- utils: 유틸리티 함수들 (NMS, stereo depth 등)
"""

# Core components
from .core import (
    ThreeTaskDSPNet,
    DSPNetBackbone,
    SSDanchorGenerator,
    create_model,
    load_pretrained_model
)

# Task-specific heads
from .heads import (
    MultiTaskDetectionHead,
    PyramidPoolingSegmentationHead,
    DepthRegressionHead,
    CrossTaskProjectionHead,
    BaseHead,
    PredictionHead,
    PyramidPoolingModule,
    conv_unit,
    cosine_similarity_loss
)

# Loss functions
from .losses import (
    FocalLoss,
    FocalSmoothL1Loss,
    CrossEntropySegmentationLoss,
    DepthLoss,
    SimpleTwoTaskLoss,
    UberNetMTPSLLoss,
    iou_match_anchors,
    box_iou,
    hard_negative_mining
)

# Data handling
from .data import (
    ThreeTaskDataset,
    ubernet_mtpsl_collate_fn,
    create_multitask_collate_fn,  # Backward compatibility
    create_dataset,
    create_ubernet_mtpsl_dataset,
    create_massive_dataset
)

# Utilities
from .utils import (
    single_class_nms,
    multiclass_nms,
    batch_nms,
    decode_ssd_predictions,
    bbox_iou,
    nms,  # Backward compatibility
    apply_nms_with_class_threshold,
    apply_nms_with_score_threshold,
    process_depth_sample,
    StereoDepthCalculator
)

__version__ = "1.0.0"
__author__ = "SafeStrp Team"

__all__ = [
    # Core
    'ThreeTaskDSPNet',
    'DSPNetBackbone', 
    'SSDanchorGenerator',
    'create_model',
    'load_pretrained_model',
    
    # Heads
    'MultiTaskDetectionHead',
    'PyramidPoolingSegmentationHead',
    'DepthRegressionHead',
    'CrossTaskProjectionHead',
    'BaseHead',
    'PredictionHead',
    'PyramidPoolingModule',
    'conv_unit',
    'cosine_similarity_loss',
    
    # Losses
    'FocalLoss',
    'FocalSmoothL1Loss',
    'CrossEntropySegmentationLoss',
    'DepthLoss',
    'SimpleTwoTaskLoss',
    'UberNetMTPSLLoss',
    'iou_match_anchors',
    'box_iou',
    'hard_negative_mining',
    
    # Data
    'ThreeTaskDataset',
    'ubernet_mtpsl_collate_fn',
    'create_multitask_collate_fn',
    'create_dataset',
    'create_ubernet_mtpsl_dataset',
    'create_massive_dataset',
    
    # Utils
    'single_class_nms',
    'multiclass_nms',
    'batch_nms',
    'decode_ssd_predictions',
    'bbox_iou',
    'nms',
    'apply_nms_with_class_threshold',
    'apply_nms_with_score_threshold',
    'process_depth_sample',
    'StereoDepthCalculator'
] 