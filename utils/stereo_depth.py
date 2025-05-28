"""
Stereo Depth Calculation Utilities

실제 SafeStrp 데이터셋의 Disparity16 + stereo calibration을 이용한 depth 계산.
각 폴더별 config 파일을 올바르게 파싱하여 정확한 depth 계산을 수행합니다.
"""

import os
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import configparser
from pathlib import Path


class StereoDepthCalculator:
    """
    SafeStrp 데이터셋용 스테레오 depth 계산기.
    
    실제 데이터 특성:
    - Disparity16: 16-bit disparity 값 (0~65535)
    - Config 파일: 각 폴더별로 다른 카메라 파라미터
    - 실제 거리: 수 미터~수십 미터 범위
    
    Depth 계산식: depth(mm) = (fx * baseline) / disparity_pixels
    """
    
    def __init__(self, 
                 fx: float = 1400.15,
                 baseline: float = 119.975,
                 max_depth_mm: float = 200000.0,  # 200m
                 min_depth_mm: float = 200.0,     # 0.2m
                 disparity_scale: float = 1.0):    # Disparity16 스케일링
        self.fx = fx
        self.baseline = baseline
        self.max_depth_mm = max_depth_mm
        self.min_depth_mm = min_depth_mm
        self.disparity_scale = disparity_scale
        
        print(f"✅ StereoDepthCalculator 초기화:")
        print(f"   fx: {fx:.2f}")
        print(f"   baseline: {baseline:.3f}mm")
        print(f"   유효 거리: {min_depth_mm/1000:.1f}m ~ {max_depth_mm/1000:.0f}m")
        print(f"   disparity scale: {disparity_scale}")
    
    @classmethod
    def from_config_file(cls, config_path: str, camera_type: str = "2K") -> 'StereoDepthCalculator':
        """
        실제 config 파일에서 카메라 파라미터를 로드하여 초기화.
        
        Args:
            config_path: Depth_XXX.conf 파일 경로
            camera_type: 카메라 해상도 ("2K", "FHD", "HD", "VGA")
        """
        if not os.path.exists(config_path):
            print(f"⚠️  Config 파일 없음: {config_path}, 기본값 사용")
            return cls()
            
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # 좌측 카메라에서 fx 가져오기
            left_cam_section = f"LEFT_CAM_{camera_type}"
            if left_cam_section not in config:
                print(f"⚠️  {left_cam_section} 섹션 없음, 기본값 사용")
                return cls()
            
            fx = float(config[left_cam_section]['fx'])
            
            # 스테레오 섹션에서 baseline 가져오기
            if "STEREO" not in config:
                print(f"⚠️  STEREO 섹션 없음, 기본값 사용")
                return cls()
            
            baseline = float(config["STEREO"]["BaseLine"])
            
            print(f"📁 Config 로드: {os.path.basename(config_path)}")
            print(f"   카메라: {camera_type}")
            
            return cls(fx=fx, baseline=baseline)
            
        except Exception as e:
            print(f"⚠️  Config 파싱 오류 ({config_path}): {e}")
            return cls()
    
    def disparity_to_depth(self, 
                          disparity16: np.ndarray, 
                          confidence_mask: Optional[np.ndarray] = None,
                          min_disparity: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Disparity16를 실제 depth로 변환.
        
        Args:
            disparity16: 16-bit disparity 이미지 (0~65535)
            confidence_mask: 신뢰도 마스크 (0~255), None이면 사용 안함
            min_disparity: 최소 유효 disparity 값
            
        Returns:
            tuple: (depth_mm, valid_mask)
                - depth_mm: mm 단위 depth 맵
                - valid_mask: 유효한 픽셀 마스크
        """
        # SafeStrp 데이터셋은 모두 256배 스케일링 사용
        scale_factor = 256.0
        
        # 실제 disparity 값으로 변환
        disparity_float = disparity16.astype(np.float32) / scale_factor
        
        print(f"   📊 Disparity 분석:")
        print(f"      원본 범위: {disparity16.min()}~{disparity16.max()}")
        print(f"      고정 스케일 팩터: {scale_factor}")
        print(f"      실제 disparity: {disparity_float.min():.2f}~{disparity_float.max():.2f}px")
        
        # 유효한 disparity 픽셀 찾기
        valid_disparity = disparity_float >= min_disparity
        
        # Confidence 마스크 적용 (있는 경우)
        if confidence_mask is not None:
            # Confidence 임계값을 낮춤 (더 많은 픽셀 포함)
            confidence_valid = confidence_mask > 50  # 255 기준으로 50 이상
            valid_mask = valid_disparity & confidence_valid
        else:
            valid_mask = valid_disparity
        
        # Depth 계산: depth(mm) = (fx * baseline) / disparity
        depth_mm = np.zeros_like(disparity_float, dtype=np.float32)
        
        if valid_mask.any():
            valid_disparity_values = disparity_float[valid_mask]
            depth_values = (self.fx * self.baseline) / valid_disparity_values
            depth_mm[valid_mask] = depth_values
            
            # 거리 범위 체크로 추가 필터링
            range_valid = (depth_mm >= self.min_depth_mm) & (depth_mm <= self.max_depth_mm)
            final_valid_mask = valid_mask & range_valid
            
            # 범위 밖 픽셀은 0으로 설정
            depth_mm[~final_valid_mask] = 0.0
            
            print(f"   💡 Depth 변환 완료:")
            print(f"      유효 픽셀: {final_valid_mask.sum():,}/{valid_mask.size:,} ({100*final_valid_mask.sum()/valid_mask.size:.1f}%)")
            if final_valid_mask.any():
                valid_depths = depth_mm[final_valid_mask]
                print(f"      거리 범위: {valid_depths.min():.1f}~{valid_depths.max():.1f}mm ({valid_depths.min()/1000:.2f}~{valid_depths.max()/1000:.2f}m)")
            
            return depth_mm, final_valid_mask
        else:
            print(f"   ⚠️  유효한 disparity 픽셀이 없습니다.")
            print(f"      디버그: disp>={min_disparity}: {valid_disparity.sum()}, conf>50: {(confidence_mask>50).sum() if confidence_mask is not None else 'N/A'}")
            return depth_mm, valid_mask
    
    def normalize_depth_for_learning(self, 
                                   depth_mm: np.ndarray, 
                                   valid_mask: np.ndarray,
                                   target_range: Tuple[float, float] = (0.0, 255.0)) -> np.ndarray:
        """
        학습용으로 depth를 정규화.
        
        Args:
            depth_mm: mm 단위 depth 맵
            valid_mask: 유효한 픽셀 마스크
            target_range: 정규화 목표 범위
            
        Returns:
            정규화된 depth 맵
        """
        normalized = np.zeros_like(depth_mm)
        
        if valid_mask.any():
            valid_depths = depth_mm[valid_mask]
            
            # Linear scaling to target range
            min_depth = valid_depths.min()
            max_depth = valid_depths.max()
            
            if max_depth > min_depth:
                # 0~1 범위로 정규화 후 target_range로 스케일링
                normalized_01 = (valid_depths - min_depth) / (max_depth - min_depth)
                target_min, target_max = target_range
                normalized_values = normalized_01 * (target_max - target_min) + target_min
                normalized[valid_mask] = normalized_values
            else:
                # 모든 값이 같은 경우 중간값 사용
                normalized[valid_mask] = (target_range[0] + target_range[1]) / 2
        
        return normalized


def process_depth_sample(image_dir: str, 
                        base_filename: str,
                        camera_type: str = "2K") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    실제 SafeStrp depth 샘플을 처리하는 헬퍼 함수.
    
    Args:
        image_dir: depth 폴더 경로 (예: data/full_dataset/depth/Depth_002)
        base_filename: 베이스 파일명 (예: ZED1_KSC_001251)
        camera_type: 카메라 타입
        
    Returns:
        tuple: (left_image, depth_mm, confidence_mask)
    """
    # 파일 경로들
    left_path = os.path.join(image_dir, f"{base_filename}_left.png")
    disp16_path = os.path.join(image_dir, f"{base_filename}_disp16.png")
    confidence_path = os.path.join(image_dir, f"{base_filename}_confidence.png")
    config_path = os.path.join(image_dir, f"{os.path.basename(image_dir)}.conf")
    
    # 파일 존재 확인
    missing_files = []
    if not os.path.exists(left_path): missing_files.append("left")
    if not os.path.exists(disp16_path): missing_files.append("disp16")
    if not os.path.exists(confidence_path): missing_files.append("confidence")
    
    if missing_files:
        raise FileNotFoundError(f"누락된 파일들: {missing_files}")
    
    # 이미지들 로드
    left_image = cv2.imread(left_path)
    disparity16 = cv2.imread(disp16_path, cv2.IMREAD_UNCHANGED)  # 16-bit 유지
    confidence = cv2.imread(confidence_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculator 초기화 (config 파일 있으면 사용)
    if os.path.exists(config_path):
        calculator = StereoDepthCalculator.from_config_file(config_path, camera_type)
    else:
        calculator = StereoDepthCalculator()
    
    # Depth 계산
    depth_mm, valid_mask = calculator.disparity_to_depth(disparity16, confidence)
    
    return left_image, depth_mm, confidence


if __name__ == "__main__":
    # 실제 데이터로 테스트
    print("🧪 실제 SafeStrp depth 데이터 테스트")
    
    test_dir = "data/full_dataset/depth/Depth_002"
    test_filename = "ZED1_KSC_001251"
    
    if os.path.exists(test_dir):
        try:
            left_img, depth_mm, conf = process_depth_sample(test_dir, test_filename)
            print(f"✅ 테스트 성공:")
            print(f"   Left 이미지: {left_img.shape}")
            print(f"   Depth 범위: {depth_mm[depth_mm>0].min():.1f}~{depth_mm[depth_mm>0].max():.1f}mm")
            print(f"   Confidence: {conf.min()}~{conf.max()}")
        except Exception as e:
            print(f"⚠️  테스트 실패: {e}")
    else:
        print("⚠️  테스트 디렉토리가 없습니다.") 