"""
ThreeTaskDSPNet - Main Model for Sidewalk Safety

Detection + Surface Segmentation + Depth Estimation을 위한 멀티태스크 모델.
기존 DSPNet 구조를 활용하여 실용적이고 효율적인 3태스크 학습을 수행합니다.

핵심 특징:
- Object Detection: 29개 장애물 클래스 (bbox만, distance 제거)
- Surface Segmentation: 7개 표면 타입 분류
- Depth Estimation: pixel-wise depth regression
- 공유 백본: ResNet-50 기반 효율적 특징 추출
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .backbone import DSPNetBackbone
from .heads import MultiTaskDetectionHead, PyramidPoolingSegmentationHead, DepthRegressionHead
from .anchors import SSDanchorGenerator


class ThreeTaskDSPNet(nn.Module):
    """
    Three-Task DSPNet for sidewalk navigation safety.
    
    Architecture:
    - Shared Backbone: ResNet-50 based feature extractor
    - Detection Head: SSD-style object detection (bbox only, no distance)
    - Surface Head: FCN-style semantic segmentation
    - Depth Head: Dense depth regression
    
    이 모델은 Detection + Surface Segmentation + Depth Estimation 3태스크를 동시에 수행합니다.
    """
    
    def __init__(self,
                 num_detection_classes: int = 29,
                 num_surface_classes: int = 7,
                 input_size: Tuple[int, int] = (512, 512),
                 pretrained_backbone: bool = True):
        """
        Initialize ThreeTaskDSPNet.
        
        Args:
            num_detection_classes: Number of object detection classes (29 obstacles)
            num_surface_classes: Number of surface segmentation classes (7 surfaces)
            input_size: Input image size (H, W)
            pretrained_backbone: Whether to use pretrained ResNet backbone
        """
        super(ThreeTaskDSPNet, self).__init__()
        
        self.num_detection_classes = num_detection_classes
        self.num_surface_classes = num_surface_classes
        self.input_size = input_size
        
        # Shared backbone for feature extraction
        self.backbone = DSPNetBackbone(pretrained=pretrained_backbone)
        
        # **NEW: SSD Anchor Generator**
        self.anchor_generator = SSDanchorGenerator(
            input_size=input_size,
            feature_scales=[8, 16, 32, 64, 128, 256, 512],  # ResNet-50 scales
            sizes=[
                [0.5, 0.705],      # Level 0: _plus6 (1/8)
                [0.1, 0.141],      # Level 1: _plus12 (1/16) 
                [0.2, 0.272],      # Level 2: _plus15 (1/32)
                [0.37, 0.447],     # Level 3: Extra layer (1/64)
                [0.54, 0.619],     # Level 4: Extra layer (1/128)
                [0.71, 0.79],      # Level 5: Extra layer (1/256)
                [0.88, 0.961]      # Level 6: Extra layer (1/512)
            ],
            aspect_ratios=[
                [1, 2, 0.5],           # Level 0: 3 ratios
                [1, 2, 0.5],           # Level 1: 3 ratios  
                [1, 2, 0.5, 3, 1./3],  # Level 2: 5 ratios
                [1, 2, 0.5, 3, 1./3],  # Level 3: 5 ratios
                [1, 2, 0.5, 3, 1./3],  # Level 4: 5 ratios
                [1, 2, 0.5],           # Level 5: 3 ratios
                [1, 2, 0.5]            # Level 6: 3 ratios
            ]
        )
        
        # Task-specific heads  
        self.detection_head = MultiTaskDetectionHead(
            num_classes=num_detection_classes,
            anchors_per_location_list=[4, 4, 6, 6, 6, 4, 4]  # DSPNet original: (len(size)-1) + len(ratio)
        )
        
        self.surface_head = PyramidPoolingSegmentationHead(
            num_classes=num_surface_classes,
            c3_channels=512,   # ResNet-50 C3 output channels
            c4_channels=1024,  # ResNet-50 C4 output channels
            c5_channels=2048   # ResNet-50 C5 output channels
        )
        
        self.depth_head = DepthRegressionHead(
            input_channels=2048,  # Assuming the last layer of ResNet-50 is 2048 channels
            output_channels=1    # Output channels for depth regression
        )
        
        # **NEW: Pre-generate anchors for efficiency**
        self._anchors = None
        self._generate_anchors()
        
        # Initialize task-specific heads
        self._initialize_weights()
        
        print(f"✅ ThreeTaskDSPNet 초기화 완료:")
        print(f"   Detection: {num_detection_classes}개 클래스 (bbox only, no distance)")
        print(f"   Surface: {num_surface_classes}개 클래스 Segmentation")
        print(f"   Depth: pixel-wise depth regression")
        print(f"   입력 크기: {input_size}")
        print(f"   **SSD Anchors**: {self._anchors.shape[0]:,}개 생성")
        print(f"   Pretrained 백본: {pretrained_backbone}")
    
    def _generate_anchors(self):
        """Pre-generate anchors for efficiency."""
        feature_maps = self.anchor_generator.get_feature_map_sizes()
        self._anchors = self.anchor_generator.generate_anchors(feature_maps)
        print(f"   🎯 Anchors 생성 완료: {self._anchors.shape}")
    
    def get_anchors(self) -> torch.Tensor:
        """Get pre-generated anchors."""
        return self._anchors
    
    def _initialize_weights(self):
        """Initialize weights for task-specific heads."""
        for module in [self.detection_head, self.surface_head, self.depth_head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the three-task network.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Dictionary containing outputs from all tasks:
            - 'detection_cls': Classification predictions (B, num_anchors, num_classes)
            - 'detection_reg': Regression predictions (B, num_anchors, 4)  # bbox(4)
            - 'surface_segmentation': Segmentation predictions (B, num_classes, H, W)
            - 'depth_estimation': Depth predictions (B, 1, H, W)
            - 'anchors': Anchor boxes (num_anchors, 4) [x1, y1, x2, y2]
        """
        # Extract multi-scale features from shared backbone
        features = self.backbone(x)  # Returns [C3, C4, C5]
        c3_feat, c4_feat, c5_feat = features[0], features[1], features[2]
        
        # Object detection (classification + bbox regression)
        detection_cls, detection_reg = self.detection_head(c3_feat, c4_feat, c5_feat)
        
        # Surface segmentation
        surface_segmentation = self.surface_head(c3_feat, c4_feat, c5_feat)
        
        # Depth estimation
        depth_estimation = self.depth_head(c5_feat)
        
        return {
            'detection_cls': detection_cls,
            'detection_reg': detection_reg,
            'surface_segmentation': surface_segmentation,
            'depth_estimation': depth_estimation,
            'anchors': self._anchors  # **NEW: Include pre-generated anchors**
        }
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate parameters for each component
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        detection_params = sum(p.numel() for p in self.detection_head.parameters())
        surface_params = sum(p.numel() for p in self.surface_head.parameters())
        depth_params = sum(p.numel() for p in self.depth_head.parameters())
        
        return {
            'model_name': 'ThreeTaskDSPNet',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'component_parameters': {
                'backbone': backbone_params,
                'detection_head': detection_params,
                'surface_head': surface_params,
                'depth_head': depth_params
            },
            'tasks': {
                'object_detection': {
                    'num_classes': self.num_detection_classes,
                    'output_format': 'SSD-style classifications + regressions (bbox)',
                    'num_anchors': 22516  # Approximate total anchors across all scales
                },
                'surface_segmentation': {
                    'num_classes': self.num_surface_classes,
                    'output_format': 'Dense pixel-wise predictions',
                    'output_size': self.input_size
                },
                'depth_estimation': {
                    'output_format': 'pixel-wise depth regression',
                    'output_size': self.input_size
                }
            },
            'input_size': self.input_size,
            'architecture': {
                'backbone': 'ResNet-50 based DSPNetBackbone',
                'detection_head': '7-level SSD pyramid',
                'surface_head': 'Pyramid pooling segmentation',
                'depth_head': 'Dense depth regression'
            }
        }
    
    def get_feature_info(self) -> Dict:
        """Get information about feature extraction."""
        return {
            'backbone_info': {
                'architecture': 'ResNet-50',
                'output_scales': ['C3 (1/8)', 'C4 (1/16)', 'C5 (1/32)'],
                'output_channels': [512, 1024, 2048]
            },
            'detection_head_info': self.detection_head.get_feature_info(),
            'surface_head_info': {
                'method': 'Pyramid Pooling',
                'scales': ['C3', 'C4', 'C5', 'Global', '2x2', '3x3', '6x6'],
                'output_scale': 'Original (1/1)'
            },
            'depth_head_info': {
                'method': 'Dense depth regression',
                'input_channels': 2048,
                'output_channels': 1
            }
        }


def create_model(config: Dict = None) -> ThreeTaskDSPNet:
    """
    Factory function to create ThreeTaskDSPNet model.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Initialized ThreeTaskDSPNet model
    """
    if config is None:
        config = {}
    
    model = ThreeTaskDSPNet(
        num_detection_classes=config.get('num_detection_classes', 29),
        num_surface_classes=config.get('num_surface_classes', 7),
        input_size=config.get('input_size', (512, 512)),
        pretrained_backbone=config.get('pretrained_backbone', True)
    )
    
    return model


def load_pretrained_model(checkpoint_path: str, device: str = 'cpu') -> ThreeTaskDSPNet:
    """
    Load pretrained ThreeTaskDSPNet model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint if available
    config = checkpoint.get('config', {})
    model = create_model(config)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Pretrained model loaded from {checkpoint_path}")
    return model


def main():
    """Test function for model creation and forward pass."""
    print("🎯 ThreeTaskDSPNet 테스트")
    print("=" * 50)
    
    # Create model
    model = ThreeTaskDSPNet(
        num_detection_classes=29,
        num_surface_classes=7
    )
    model.eval()
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 512)
    
    print(f"\n🔍 Forward Pass 테스트:")
    with torch.no_grad():
        outputs = model(dummy_input)
        
        print(f"   입력 형태: {dummy_input.shape}")
        print(f"   Detection 출력:")
        print(f"      Classification: {outputs['detection_cls'].shape}")
        print(f"      Regression (bbox): {outputs['detection_reg'].shape}")
        print(f"   Surface 출력: {outputs['surface_segmentation'].shape}")
        print(f"   Depth 출력: {outputs['depth_estimation'].shape}")
    
    # Print model information
    info = model.get_model_info()
    print(f"\n📊 모델 정보:")
    print(f"   전체 파라미터: {info['total_parameters']:,}개")
    print(f"   백본: {info['component_parameters']['backbone']:,}개")
    print(f"   Detection: {info['component_parameters']['detection_head']:,}개")
    print(f"   Surface: {info['component_parameters']['surface_head']:,}개")
    print(f"   Depth: {info['component_parameters']['depth_head']:,}개")
    
    print(f"\n✅ 모델 테스트 완료!")


if __name__ == "__main__":
    main() 