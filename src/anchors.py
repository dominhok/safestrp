"""
Anchor Generation for SSD-style Detection

원본 DSPNet을 참고한 정확한 anchor generation 구현.
ResNet-50 기반으로 7개 레벨의 feature map에서 anchor를 생성합니다.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import math


class SSDanchorGenerator:
    """
    SSD 스타일 Anchor Generator (원본 DSPNet 기반).
    
    원본 설정:
    - ResNet-50: 7개 레벨 (_plus6, _plus12, _plus15, extra layers)
    - sizes: [[.5, .705], [.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
    - ratios: [[1,2,.5], [1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
    """
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (512, 512),
                 feature_scales: List[int] = [8, 16, 32, 64, 128, 256, 512],  # ResNet-50 기반
                 sizes: Optional[List[List[float]]] = None,
                 aspect_ratios: Optional[List[List[float]]] = None,
                 clip: bool = True):
        """
        Initialize SSD anchor generator based on original DSPNet.
        
        Args:
            input_size: Input image size (H, W)
            feature_scales: Feature map scales relative to input [8, 16, 32, 64, 128, 256, 512]
            sizes: Anchor sizes for each feature level (original DSPNet format)
            aspect_ratios: Aspect ratios for each feature level
            clip: Whether to clip anchors to image boundaries
        """
        self.input_size = input_size
        self.feature_scales = feature_scales
        self.clip = clip
        
        # 원본 DSPNet의 정확한 설정 사용
        if sizes is None:
            self.sizes = [
                [0.5, 0.705],      # Level 0: _plus6 (1/8)
                [0.1, 0.141],      # Level 1: _plus12 (1/16) 
                [0.2, 0.272],      # Level 2: _plus15 (1/32)
                [0.37, 0.447],     # Level 3: Extra layer (1/64)
                [0.54, 0.619],     # Level 4: Extra layer (1/128)
                [0.71, 0.79],      # Level 5: Extra layer (1/256)
                [0.88, 0.961]      # Level 6: Extra layer (1/512)
            ]
        else:
            self.sizes = sizes
            
        if aspect_ratios is None:
            self.aspect_ratios = [
                [1, 2, 0.5],           # Level 0: 3 ratios
                [1, 2, 0.5],           # Level 1: 3 ratios  
                [1, 2, 0.5, 3, 1./3],  # Level 2: 5 ratios
                [1, 2, 0.5, 3, 1./3],  # Level 3: 5 ratios
                [1, 2, 0.5, 3, 1./3],  # Level 4: 5 ratios
                [1, 2, 0.5],           # Level 5: 3 ratios
                [1, 2, 0.5]            # Level 6: 3 ratios
            ]
        else:
            self.aspect_ratios = aspect_ratios
        
        assert len(self.sizes) == len(self.aspect_ratios) == len(self.feature_scales), \
            "sizes, aspect_ratios, and feature_scales must have the same length"
        
        # 계산된 앵커 수
        self.anchors_per_location = []
        for i, (size_list, ratio_list) in enumerate(zip(self.sizes, self.aspect_ratios)):
            # 원본 DSPNet 공식: num_anchors = len(size) - 1 + len(ratio)
            num_anchors = len(size_list) - 1 + len(ratio_list)
            self.anchors_per_location.append(num_anchors)
        
        print(f"✅ SSD Anchor Generator 초기화 (원본 DSPNet 설정):")
        print(f"   Feature scales: {self.feature_scales}")
        print(f"   Anchors per location: {self.anchors_per_location}")
        print(f"   Total levels: {len(self.feature_scales)}")
    
    def get_feature_map_sizes(self) -> List[Tuple[int, int]]:
        """Get feature map sizes for each level."""
        H, W = self.input_size
        feature_map_sizes = []
        
        for scale in self.feature_scales:
            fh, fw = H // scale, W // scale
            feature_map_sizes.append((fh, fw))
        
        return feature_map_sizes
    
    def generate_single_level_anchors(self, 
                                    feature_size: Tuple[int, int],
                                    level: int,
                                    device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Generate anchors for a single feature level (원본 DSPNet 방식).
        
        Args:
            feature_size: Feature map size (H, W)
            level: Feature level index
            device: Device to create tensors on
            
        Returns:
            Anchors tensor (num_anchors, 4) [x1, y1, x2, y2]
        """
        fh, fw = feature_size
        scale = self.feature_scales[level]
        sizes = self.sizes[level]
        ratios = self.aspect_ratios[level]
        
        # Step size (pixel spacing between anchor centers)
        step_h = self.input_size[0] / fh
        step_w = self.input_size[1] / fw
        
        # 원본 DSPNet anchor generation 로직
        anchors = []
        
        for y in range(fh):
            for x in range(fw):
                # Anchor center
                cx = (x + 0.5) * step_w
                cy = (y + 0.5) * step_h
                
                # 원본 공식: num_anchors = len(size) - 1 + len(ratio)
                # First: min_size with ratios
                min_size = sizes[0] * min(self.input_size)
                
                for ratio in ratios:
                    w = min_size * math.sqrt(ratio)
                    h = min_size / math.sqrt(ratio)
                    
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([x1, y1, x2, y2])
                
                # Second: max_size with ratio=1 (if max_size exists)
                if len(sizes) > 1:
                    max_size = sizes[1] * min(self.input_size)
                    w = h = math.sqrt(min_size * max_size)
                    
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([x1, y1, x2, y2])
        
        anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
        
        # Clip anchors to image boundaries
        if self.clip:
            anchors[:, 0::2] = torch.clamp(anchors[:, 0::2], 0, self.input_size[1])  # x coordinates
            anchors[:, 1::2] = torch.clamp(anchors[:, 1::2], 0, self.input_size[0])  # y coordinates
        
        return anchors
    
    def generate_anchors(self, 
                        feature_map_sizes: Optional[List[Tuple[int, int]]] = None,
                        device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Generate all anchors for all feature levels.
        
        Args:
            feature_map_sizes: Feature map sizes, if None will compute from feature_scales
            device: Device to create tensors on
            
        Returns:
            All anchors (total_anchors, 4) [x1, y1, x2, y2]
        """
        if feature_map_sizes is None:
            feature_map_sizes = self.get_feature_map_sizes()
        
        all_anchors = []
        
        for level, feature_size in enumerate(feature_map_sizes):
            level_anchors = self.generate_single_level_anchors(feature_size, level, device)
            all_anchors.append(level_anchors)
            
            print(f"   Level {level}: {feature_size} -> {level_anchors.shape[0]:,} anchors")
        
        # Concatenate all anchors
        anchors = torch.cat(all_anchors, dim=0)
        
        print(f"✅ Total anchors generated: {anchors.shape[0]:,}")
        return anchors
    
    def get_anchors_per_location_list(self) -> List[int]:
        """Get number of anchors per location for each level."""
        return self.anchors_per_location.copy()


def test_anchor_generator():
    """Test anchor generator with original DSPNet configuration."""
    print("🧪 DSPNet Anchor Generator 테스트")
    print("=" * 50)
    
    # Create anchor generator with original settings
    anchor_gen = SSDanchorGenerator(
        input_size=(512, 512),
        feature_scales=[8, 16, 32, 64, 128, 256, 512],  # ResNet-50 scales
    )
    
    # Generate anchors
    feature_map_sizes = anchor_gen.get_feature_map_sizes()
    print(f"\n📐 Feature map sizes: {feature_map_sizes}")
    
    anchors = anchor_gen.generate_anchors(feature_map_sizes)
    
    print(f"\n📊 Anchor 통계:")
    print(f"   총 앵커 수: {anchors.shape[0]:,}개")
    print(f"   앵커 형태: {anchors.shape}")
    print(f"   좌표 범위: x=[{anchors[:, 0].min():.1f}, {anchors[:, 2].max():.1f}], "
          f"y=[{anchors[:, 1].min():.1f}, {anchors[:, 3].max():.1f}]")
    
    # 각 레벨별 앵커 수 확인
    print(f"\n🎯 레벨별 앵커 분포:")
    total_check = 0
    for i, (fh, fw) in enumerate(feature_map_sizes):
        expected_anchors = fh * fw * anchor_gen.anchors_per_location[i]
        total_check += expected_anchors
        print(f"   Level {i}: {fh}x{fw} x {anchor_gen.anchors_per_location[i]} = {expected_anchors:,}")
    
    print(f"\n✅ 검증: 예상 {total_check:,} vs 실제 {anchors.shape[0]:,}")
    
    # Anchor 크기 분포 확인
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    areas = widths * heights
    
    print(f"\n📏 Anchor 크기 분포:")
    print(f"   Width: [{widths.min():.1f}, {widths.max():.1f}], 평균: {widths.mean():.1f}")
    print(f"   Height: [{heights.min():.1f}, {heights.max():.1f}], 평균: {heights.mean():.1f}")
    print(f"   Area: [{areas.min():.1f}, {areas.max():.1f}], 평균: {areas.mean():.1f}")
    
    return anchors


if __name__ == "__main__":
    test_anchor_generator() 