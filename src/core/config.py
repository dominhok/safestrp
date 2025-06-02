"""
SafeStrp 모델 설정 파일

모든 모델 구성 요소의 기본 설정값을 중앙에서 관리합니다.
"""

# 모델 기본 설정
MODEL_CONFIG = {
    'backbone': 'resnet50',
    'num_detection_classes': 29,
    'num_surface_classes': 7,
    'enable_cross_task_consistency': True,
    'input_size': (512, 512),
    'pretrained_backbone': True
}

# Detection 설정
DETECTION_CONFIG = {
    'anchors_per_location_list': [4, 4, 6, 6, 6, 4, 4],
    'feature_scales': [8, 16, 32, 64, 128, 256, 512],
    'min_filter_extra_layers': 128,
    'num_feature_levels': 7
}

# Segmentation 설정
SEGMENTATION_CONFIG = {
    'num_classes': 7,
    'ignore_index': 255,
    'pyramid_pooling_sizes': [1, 2, 3, 6],
    'pooling_output_channels': 128
}

# Depth 설정
DEPTH_CONFIG = {
    'output_channels': 1,
    'skip_connections': True,
    'upsampling_mode': 'bilinear',
    'activation': 'relu'
}

# Loss 설정
LOSS_CONFIG = {
    'detection_weight': 1.0,
    'surface_weight': 0.5,  # 낮춤 (surface loss가 컸음)
    'depth_weight': 1.0,
    'cross_task_weight': 0.1,  # 낮춤
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'regularization_weight': 0.1
}

# Anchor 설정
ANCHOR_CONFIG = {
    'min_sizes': [30, 60, 111, 162, 213, 264, 315],
    'max_sizes': [60, 111, 162, 213, 264, 315, 366],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'offset': 0.5
}

# 훈련 설정
TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'num_epochs': 100,
    'warmup_epochs': 5,
    'scheduler': 'cosine'
}

# 데이터 설정  
DATA_CONFIG = {
    'image_size': (512, 512),
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'num_workers': 4,
    'pin_memory': True
} 