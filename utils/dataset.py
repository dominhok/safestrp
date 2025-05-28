"""
Dataset Utilities for TwoTaskDSPNet

Detection + Surface 2태스크를 위한 데이터셋 처리 유틸리티.
기존 3태스크 코드를 단순화하여 필수 기능만 제공합니다.

추가: Stereo depth 계산 지원
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from collections import defaultdict
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Stereo depth calculation
try:
    from utils.stereo_depth import StereoDepthCalculator
except ImportError:
    try:
        from .stereo_depth import StereoDepthCalculator
    except ImportError:
        print("⚠️  StereoDepthCalculator import 실패, 기본 depth 계산 사용")
        StereoDepthCalculator = None

# Surface 클래스 정의 (XML에서 추출한 라벨 기반)
SURFACE_LABELS = [
    'background',         # 0
    'alley',             # 1
    'bike_lane',         # 2
    'braille_guide_blocks',  # 3
    'caution_zone',      # 4
    'roadway',           # 5
    'sidewalk'           # 6
]

SURFACE_LABEL_TO_ID = {label: idx for idx, label in enumerate(SURFACE_LABELS)}

# Detection 클래스 (29개 - 실제 XML에서 확인된 모든 라벨)
DETECTION_CLASSES = [
    'barricade', 'bench', 'bicycle', 'bollard', 'bus', 'car', 'carrier', 'cat',
    'chair', 'dog', 'fire_hydrant', 'kiosk', 'motorcycle', 'movable_signage', 
    'parking_meter', 'person', 'pole', 'potted_plant', 'power_controller', 
    'scooter', 'stop', 'stroller', 'table', 'traffic_light', 'traffic_light_controller',
    'traffic_sign', 'tree_trunk', 'truck', 'wheelchair'
]

# Depth 클래스 정의 (단순한 거리 회귀)
DEPTH_RANGE = (0.0, 255.0)  # 미터 단위

class ThreeTaskDataset(Dataset):
    """
    Dataset for Detection + Surface + Depth three-task learning.
    
    각 태스크는 완전히 독립적으로 처리:
    - Detection: bbox 예측 (객체별 5차원)
    - Surface: pixel-wise segmentation (7클래스)  
    - Depth: pixel-wise regression (연속값)
    """
    
    def __init__(self,
                 base_dir: str = "data/original_dataset",
                 mode: str = "train",
                 train_ratio: float = 0.8,
                 max_samples: int = 2000,
                 input_size: Tuple[int, int] = (512, 512)):
        """
        Initialize three-task dataset.
        
        Args:
            base_dir: Path to original dataset
            mode: 'train' or 'val'
            train_ratio: Train/validation split ratio
            max_samples: Maximum number of samples to load
            input_size: Input image size (H, W)
        """
        self.base_dir = base_dir
        self.mode = mode
        self.input_size = input_size
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and split data
        print("🔍 3개 태스크 데이터 로딩 중...")
        self.detection_data = self._load_detection_data()[:max_samples//3]
        self.surface_data = self._load_surface_data()[:max_samples//3] 
        self.depth_data = self._load_depth_data()[:max_samples//3]
        
        self._split_data(train_ratio)
        self._create_sample_list()
        
        print(f"✅ ThreeTaskDataset ({mode}) 생성:")
        print(f"   Detection: {len(self.final_detection)}개 (bbox 예측)")
        print(f"   Surface: {len(self.final_surface)}개 (pixel segmentation)")
        print(f"   Depth: {len(self.final_depth)}개 (pixel regression)")
        print(f"   전체: {len(self.samples)}개")
    
    def _load_detection_data(self) -> List[Dict]:
        """Load detection data."""
        detection_dir = os.path.join(self.base_dir, "bbox")
        data = []
        
        if not os.path.exists(detection_dir):
            print(f"⚠️  Detection 디렉토리가 없습니다: {detection_dir}")
            return data
        
        subfolders = [d for d in os.listdir(detection_dir) 
                     if os.path.isdir(os.path.join(detection_dir, d))]
        print(f"🔍 Detection 폴더 발견: {len(subfolders)}개")
        
        for subfolder in subfolders:  # 모든 폴더 사용
            subfolder_path = os.path.join(detection_dir, subfolder)
            xml_files = [f for f in os.listdir(subfolder_path) if f.endswith('.xml')]
            print(f"   📁 {subfolder}: XML={len(xml_files)}개")
            
            if xml_files:
                xml_path = os.path.join(subfolder_path, xml_files[0])
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    image_count = 0
                    for image_elem in root.findall('image'):  # 모든 이미지 사용
                        image_name = image_elem.get('name')
                        if image_name:
                            image_path = os.path.join(subfolder_path, image_name)
                            if os.path.exists(image_path):
                                boxes = []
                                for box in image_elem.findall('box'):
                                    label = box.get('label')
                                    if label in DETECTION_CLASSES:
                                        try:
                                            xtl = float(box.get('xtl'))
                                            ytl = float(box.get('ytl'))
                                            xbr = float(box.get('xbr'))
                                            ybr = float(box.get('ybr'))
                                            class_id = DETECTION_CLASSES.index(label)
                                            boxes.append([xtl, ytl, xbr, ybr, class_id])
                                        except:
                                            continue
                                
                                if boxes:
                                    data.append({
                                        'image_path': image_path,
                                        'boxes': boxes,
                                        'task': 'detection'
                                    })
                                    image_count += 1
                    print(f"      → {image_count}개 이미지 로드됨")
                except Exception as e:
                    print(f"⚠️  {subfolder} XML 파싱 오류: {e}")
                    continue
        
        print(f"✅ 총 Detection 데이터: {len(data)}개")
        return data
    
    def _load_surface_data(self) -> List[Dict]:
        """Load surface data with proper XML structure understanding."""
        surface_dir = os.path.join(self.base_dir, "surface")
        data = []
        
        if not os.path.exists(surface_dir):
            print(f"⚠️  Surface 디렉토리가 없습니다: {surface_dir}")
            return data
        
        subfolders = [d for d in os.listdir(surface_dir) 
                     if os.path.isdir(os.path.join(surface_dir, d))]
        print(f"🔍 Surface 폴더 발견: {len(subfolders)}개")
        
        for subfolder in subfolders:  # 모든 폴더 사용
            subfolder_path = os.path.join(surface_dir, subfolder)
            mask_dir = os.path.join(subfolder_path, 'MASK')
            
            # XML 파일 찾기 (폴더당 하나)
            xml_files = [f for f in os.listdir(subfolder_path) if f.endswith('.xml')]
            print(f"   📁 {subfolder}: XML={len(xml_files)}개, MASK 폴더={'존재' if os.path.exists(mask_dir) else '없음'}")
            
            if not xml_files or not os.path.exists(mask_dir):
                continue
                
            xml_path = os.path.join(subfolder_path, xml_files[0])
            
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                image_count = 0
                # 각 이미지별로 처리 (XML 구조 올바르게 이해)
                for image_elem in root.findall('image'):
                    image_name = image_elem.get('name')
                    if not image_name:
                        continue
                    
                    # 이미지 파일과 마스크 파일 경로
                    image_path = os.path.join(subfolder_path, image_name)
                    mask_name = os.path.splitext(image_name)[0] + '.png'
                    mask_path = os.path.join(mask_dir, mask_name)
                    
                    if os.path.exists(image_path) and os.path.exists(mask_path):
                        # 이 이미지에서 실제 사용된 라벨들 추출
                        used_labels = set()
                        label_attributes = {}
                        
                        for polygon in image_elem.findall('polygon'):
                            label = polygon.get('label')
                            if label:
                                used_labels.add(label)
                                
                                # 속성도 추출
                                attr_elem = polygon.find('attribute[@name="attribute"]')
                                if attr_elem is not None and attr_elem.text:
                                    label_attributes[label] = attr_elem.text
                        
                        data.append({
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'task': 'surface',
                            'used_labels': list(used_labels),  # 실제 사용된 라벨들
                            'label_attributes': label_attributes  # 라벨별 속성
                        })
                        image_count += 1
                        
                print(f"      → {image_count}개 이미지 로드됨")
                        
            except Exception as e:
                print(f"⚠️  {subfolder} XML 파싱 오류: {e}")
                continue
        
        print(f"✅ 총 Surface 데이터: {len(data)}개")
        return data
    
    def _load_depth_data(self) -> List[Dict]:
        """Load depth data with proper stereo depth structure."""
        depth_dir = os.path.join(self.base_dir, "depth")
        data = []
        
        if not os.path.exists(depth_dir):
            print(f"⚠️  Depth 디렉토리가 없습니다: {depth_dir}")
            return data
        
        subfolders = [d for d in os.listdir(depth_dir) 
                     if os.path.isdir(os.path.join(depth_dir, d))]
        print(f"🔍 Depth 폴더 발견: {len(subfolders)}개")
        
        for subfolder in subfolders:  # 모든 폴더 사용
            subfolder_path = os.path.join(depth_dir, subfolder)
            
            # 각 폴더에서 실제 파일 패턴에 맞는 파일들 찾기
            files = os.listdir(subfolder_path)
            
            # 실제 파일 패턴: ZED1_KSC_XXXXXX_left.png, ZED1_KSC_XXXXXX_disp16.png, ZED1_KSC_XXXXXX_confidence.png
            left_images = [f for f in files if f.endswith('_left.png')]  # Raw_Left → _left.png
            disp16_files = [f for f in files if f.endswith('_disp16.png')]  # Disparity16
            confidence_files = [f for f in files if f.endswith('_confidence.png') and '_confidence_save.png' not in f]  # Confidence (save 제외)
            
            print(f"   📁 {subfolder}: Left={len(left_images)}개, Disp16={len(disp16_files)}개, Conf={len(confidence_files)}개")
            
            # 각 이미지에 대해 매칭
            image_count = 0
            for left_img in left_images:
                # 파일명에서 베이스 이름 추출 (예: ZED1_KSC_001251_left.png → ZED1_KSC_001251)
                base_name = left_img.replace('_left.png', '')
                
                # 매칭되는 disparity와 confidence 파일 찾기
                disp16_file = f"{base_name}_disp16.png"
                confidence_file = f"{base_name}_confidence.png"
                
                left_path = os.path.join(subfolder_path, left_img)
                disp16_path = os.path.join(subfolder_path, disp16_file)
                confidence_path = os.path.join(subfolder_path, confidence_file)
                
                # 모든 파일이 존재하는지 확인
                if (os.path.exists(left_path) and 
                    os.path.exists(disp16_path) and 
                    os.path.exists(confidence_path)):
                    
                    data.append({
                        'image_path': left_path,        # Left 이미지 (입력)
                        'disparity_path': disp16_path,  # Disparity16 (depth 정보)
                        'confidence_path': confidence_path,  # Confidence (신뢰도)
                        'base_name': base_name,         # 베이스 파일명
                        'task': 'depth'
                    })
                    image_count += 1
                else:
                    # 디버깅: 누락된 파일 확인
                    missing = []
                    if not os.path.exists(left_path): missing.append('left')
                    if not os.path.exists(disp16_path): missing.append('disp16')
                    if not os.path.exists(confidence_path): missing.append('confidence')
                    if missing:
                        print(f"      ⚠️  {base_name}: 누락 파일 = {missing}")
                    
            print(f"      → {image_count}개 depth 샘플 로드됨")
        
        print(f"✅ 총 Depth 데이터: {len(data)}개")
        return data
    
    def _split_data(self, train_ratio: float):
        """Split data into train/validation."""
        # Detection split
        random.shuffle(self.detection_data)
        split_idx = int(len(self.detection_data) * train_ratio)
        
        if self.mode == 'train':
            self.final_detection = self.detection_data[:split_idx]
        else:
            self.final_detection = self.detection_data[split_idx:]
        
        # Surface split
        random.shuffle(self.surface_data)
        split_idx = int(len(self.surface_data) * train_ratio)
        
        if self.mode == 'train':
            self.final_surface = self.surface_data[:split_idx]
        else:
            self.final_surface = self.surface_data[split_idx:]
        
        # Depth split
        random.shuffle(self.depth_data)
        split_idx = int(len(self.depth_data) * train_ratio)
        
        if self.mode == 'train':
            self.final_depth = self.depth_data[:split_idx]
        else:
            self.final_depth = self.depth_data[split_idx:]
    
    def _create_sample_list(self):
        """Create unified sample list."""
        self.samples = []
        
        # Add detection samples
        for item in self.final_detection:
            self.samples.append(('detection', item))
        
        # Add surface samples
        for item in self.final_surface:
            self.samples.append(('surface', item))
        
        # Add depth samples
        for item in self.final_depth:
            self.samples.append(('depth', item))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        task_type, item = self.samples[idx]
        
        if task_type == 'detection':
            return self._get_detection_sample(item)
        elif task_type == 'surface':
            return self._get_surface_sample(item)
        elif task_type == 'depth':
            return self._get_depth_sample(item)
    
    def _get_detection_sample(self, item: Dict) -> Dict:
        """Get detection sample without depth processing."""
        try:
            image = Image.open(item['image_path']).convert('RGB')
            image_tensor = self.transform(image)
            
            # 원본 이미지 크기
            original_width, original_height = image.size
            
            # 간단한 bbox 처리 (depth는 제거)
            for bbox_data in item['boxes']:
                # 5차원 유지: [x1, y1, x2, y2, class_id]
                # depth 관련 distance는 제거
                pass
            
            return {
                'image': image_tensor,
                'task': 'detection',
                'boxes': torch.tensor(item['boxes'], dtype=torch.float32),  # 5차원: [x1,y1,x2,y2,cls]
                'has_detection': True,
                'has_surface': False
            }
        except Exception as e:
            print(f"⚠️  Detection 샘플 처리 중 오류: {e}")
            # Return dummy sample on error
            return {
                'image': torch.zeros(3, *self.input_size),
                'task': 'detection',
                'boxes': torch.zeros(1, 5),  # 5차원으로 변경
                'has_detection': True,
                'has_surface': False
            }
    
    def _get_surface_sample(self, item: Dict) -> Dict:
        """Get surface sample using XML polygon data."""
        try:
            # 이미지 로드
            image = Image.open(item['image_path']).convert('RGB')
            original_width, original_height = image.size
            
            # XML에서 폴리곤 정보 추출
            surface_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            
            # XML 파일 경로 추출
            xml_path = None
            for file in os.listdir(os.path.dirname(item['image_path'])):
                if file.endswith('.xml'):
                    xml_path = os.path.join(os.path.dirname(item['image_path']), file)
                    break
            
            if xml_path and os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    # 현재 이미지의 폴리곤들 찾기
                    image_name = os.path.basename(item['image_path'])
                    for image_elem in root.findall('image'):
                        if image_elem.get('name') == image_name:
                            # 이 이미지의 모든 폴리곤 처리
                            for polygon in image_elem.findall('polygon'):
                                label = polygon.get('label')
                                points_str = polygon.get('points')
                                
                                if label in SURFACE_LABEL_TO_ID and points_str:
                                    class_id = SURFACE_LABEL_TO_ID[label]
                                    
                                    # 포인트 파싱 (x1,y1;x2,y2;... 형태)
                                    try:
                                        points = []
                                        for point_pair in points_str.split(';'):
                                            if ',' in point_pair:
                                                x, y = point_pair.split(',')
                                                points.append([int(float(x)), int(float(y))])
                                        
                                        if len(points) >= 3:  # 최소 3개 점이 있어야 폴리곤
                                            # 폴리곤을 마스크에 그리기
                                            cv2.fillPoly(surface_mask, [np.array(points)], class_id)
                                    except:
                                        continue
                            break
                except Exception as e:
                    print(f"⚠️  XML 파싱 오류: {e}")
            
            # 마스크 리사이즈
            surface_mask = cv2.resize(surface_mask, self.input_size, interpolation=cv2.INTER_NEAREST)
            
            # 이미지 변환
            image_tensor = self.transform(image)
            
            return {
                'image': image_tensor,
                'task': 'surface',
                'surface_mask': torch.tensor(surface_mask, dtype=torch.long),
                'has_detection': False,
                'has_surface': True
            }
            
        except Exception as e:
            print(f"⚠️  Surface 샘플 로딩 오류: {e}")
            # Return dummy sample on error
            return {
                'image': torch.zeros(3, *self.input_size),
                'task': 'surface',
                'surface_mask': torch.zeros(*self.input_size, dtype=torch.long),
                'has_detection': False,
                'has_surface': True
            }
    
    def _get_depth_sample(self, item: Dict) -> Dict:
        """Get depth sample with proper SafeStrp stereo depth processing."""
        try:
            # Left 이미지 로드 (입력 이미지)
            image = Image.open(item['image_path']).convert('RGB')
            image_tensor = self.transform(image)
            
            # Disparity16와 Confidence 로드 (16-bit 유지)
            disparity16 = cv2.imread(item['disparity_path'], cv2.IMREAD_UNCHANGED)
            confidence = cv2.imread(item['confidence_path'], cv2.IMREAD_GRAYSCALE)
            
            if disparity16 is None:
                raise ValueError(f"Disparity16 로드 실패: {item['disparity_path']}")
            if confidence is None:
                raise ValueError(f"Confidence 로드 실패: {item['confidence_path']}")
            
            # StereoDepthCalculator 사용하여 정확한 depth 계산
            folder_path = os.path.dirname(item['disparity_path'])
            config_files = [f for f in os.listdir(folder_path) if f.endswith('.conf')]
            
            if StereoDepthCalculator is not None:
                # Config 파일이 있으면 사용, 없으면 기본값
                if config_files:
                    config_path = os.path.join(folder_path, config_files[0])
                    try:
                        calculator = StereoDepthCalculator.from_config_file(config_path, camera_type="2K")
                    except Exception as e:
                        print(f"   ⚠️  Config 로드 실패 ({item['base_name']}): {e}")
                        calculator = StereoDepthCalculator()
                else:
                    calculator = StereoDepthCalculator()
                
                # Disparity16를 실제 depth(mm)로 변환
                depth_mm, valid_mask = calculator.disparity_to_depth(disparity16, confidence)
                
                # 학습용으로 0-255 범위로 정규화
                if valid_mask.any():
                    depth_normalized = calculator.normalize_depth_for_learning(
                        depth_mm, valid_mask, target_range=(0.0, 255.0)
                    )
                else:
                    print(f"   ⚠️  유효한 depth 픽셀이 없음 ({item['base_name']})")
                    depth_normalized = np.zeros_like(depth_mm)
                    
            else:
                # Fallback: 간단한 depth 계산 (StereoDepthCalculator import 실패 시)
                print(f"   ⚠️  StereoDepthCalculator 없음, fallback 사용 ({item['base_name']})")
                focal_length = 1400.15
                baseline = 119.975
                
                disparity_float = disparity16.astype(np.float32)
                valid_mask = (disparity_float > 1.0) & (confidence > 0)
                
                depth_mm = np.zeros_like(disparity_float)
                if valid_mask.any():
                    depth_mm[valid_mask] = (baseline * focal_length) / disparity_float[valid_mask]
                    depth_mm = np.clip(depth_mm, 200, 200000)  # 0.2m ~ 200m
                
                # 간단한 정규화
                if valid_mask.any():
                    valid_depths = depth_mm[valid_mask]
                    min_d, max_d = valid_depths.min(), valid_depths.max()
                    depth_normalized = np.zeros_like(depth_mm)
                    if max_d > min_d:
                        depth_normalized[valid_mask] = ((valid_depths - min_d) / (max_d - min_d)) * 255.0
                    else:
                        depth_normalized[valid_mask] = 127.5
                else:
                    depth_normalized = np.zeros_like(depth_mm)
            
            # Resize to target input size
            depth_resized = cv2.resize(depth_normalized, self.input_size, interpolation=cv2.INTER_NEAREST)
            confidence_resized = cv2.resize(confidence / 255.0, self.input_size, interpolation=cv2.INTER_NEAREST)
            
            return {
                'image': image_tensor,
                'depth_map': torch.tensor(depth_resized, dtype=torch.float32),
                'confidence_mask': torch.tensor(confidence_resized, dtype=torch.float32),
                'task': 'depth',
                'has_detection': False,
                'has_surface': False,
                'base_name': item['base_name']
            }
            
        except Exception as e:
            print(f"⚠️  Depth 샘플 처리 오류 ({item.get('base_name', 'unknown')}): {e}")
            return {
                'image': torch.zeros(3, *self.input_size),
                'depth_map': torch.zeros(self.input_size, dtype=torch.float32),
                'confidence_mask': torch.zeros(self.input_size, dtype=torch.float32),
                'task': 'depth',
                'has_detection': False,
                'has_surface': False,
                'base_name': item.get('base_name', 'error')
            }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for Detection + Surface + Depth tasks."""
    images = torch.stack([item['image'] for item in batch])
    tasks = [item['task'] for item in batch]
    has_detection = [item.get('has_detection', False) for item in batch]
    has_surface = [item.get('has_surface', False) for item in batch]
    has_depth = [item.get('task') == 'depth' for item in batch]
    
    # Collect surface masks
    surface_masks = []
    for item in batch:
        if 'surface_mask' in item:
            surface_masks.append(item['surface_mask'])
        else:
            surface_masks.append(torch.zeros(512, 512, dtype=torch.long))
    
    # Collect detection data and separate bbox from labels (SSD 구조에 맞게)
    detection_boxes = []  # bbox 좌표만 (N, 4)
    detection_labels = []  # 클래스 라벨만 (N,)
    
    for item in batch:
        if 'boxes' in item and item['boxes'].size(0) > 0:
            boxes_data = item['boxes']  # (N, 5) - [x1, y1, x2, y2, class]
            
            # bbox 좌표 (4차원)와 클래스 라벨 분리
            bbox_coords = boxes_data[:, :4]  # (N, 4) - [x1, y1, x2, y2]
            labels = boxes_data[:, 4].long()  # (N,) - class_id
            
            detection_boxes.append(bbox_coords)
            detection_labels.append(labels)
        else:
            # 빈 박스 처리
            detection_boxes.append(torch.zeros(1, 4, dtype=torch.float32))
            detection_labels.append(torch.zeros(1, dtype=torch.long))
    
    # Collect depth data
    depth_maps = []
    confidence_masks = []
    for item in batch:
        if 'depth_map' in item:
            depth_maps.append(item['depth_map'])
            confidence_masks.append(item['confidence_mask'])
        else:
            depth_maps.append(torch.zeros(512, 512, dtype=torch.float32))
            confidence_masks.append(torch.zeros(512, 512, dtype=torch.float32))
    
    return {
        'images': images,
        'tasks': tasks,
        'has_detection': has_detection,
        'has_surface': has_surface,
        'has_depth': has_depth,
        'surface_masks': torch.stack(surface_masks),
        'detection_boxes': detection_boxes,    # List[Tensor(N, 4)] - bbox 좌표만
        'detection_labels': detection_labels,  # List[Tensor(N,)] - 클래스 라벨만  
        'depth_maps': torch.stack(depth_maps),
        'confidence_masks': torch.stack(confidence_masks),
    }


def create_dataset(base_dir: str = "data/original_dataset", 
                  batch_size: int = 8, 
                  num_workers: int = 4, 
                  max_samples: int = 2000) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        base_dir: Path to dataset
        batch_size: Batch size
        num_workers: Number of workers
        max_samples: Maximum samples to load
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print(f"🔧 데이터셋 생성 중...")
    
    train_dataset = ThreeTaskDataset(
        base_dir=base_dir,
        mode='train', 
        max_samples=max_samples
    )
    val_dataset = ThreeTaskDataset(
        base_dir=base_dir,
        mode='val', 
        max_samples=max_samples//4
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False,  # 메모리 문제 해결을 위해 비활성화
        drop_last=True,
        collate_fn=collate_fn  # 커스텀 collate 함수 사용
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False,  # 메모리 문제 해결을 위해 비활성화
        drop_last=False,
        collate_fn=collate_fn  # 커스텀 collate 함수 사용
    )
    
    return train_loader, val_loader


def create_massive_dataset(base_dir: str = "data/15.인도보행영상", 
                          batch_size: int = 24, 
                          num_workers: int = 0,  # multiprocessing 문제 해결을 위해 0으로 설정
                          max_samples: int = 5000000) -> Tuple[DataLoader, DataLoader]:
    """
    Create massive dataset loaders for full-scale training.
    
    Args:
        base_dir: Path to massive dataset (15.인도보행영상)
        batch_size: Large batch size for full training
        num_workers: More workers for data loading
        max_samples: Very large sample count
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print(f"🚀 대규모 데이터셋 생성 중...")
    
    train_dataset = ThreeTaskDataset(
        base_dir=base_dir,
        mode='train', 
        max_samples=max_samples
    )
    val_dataset = ThreeTaskDataset(
        base_dir=base_dir,
        mode='val', 
        max_samples=max_samples//10  # 검증 데이터는 10%
    )
    
    # persistent_workers는 num_workers > 0일 때만 사용
    use_persistent_workers = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent_workers,  # 조건부 활성화
        collate_fn=collate_fn  # 커스텀 collate 함수 사용
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size//2,  # 검증은 절반 배치 크기
        shuffle=False, 
        num_workers=max(1, num_workers//2) if num_workers > 0 else 0,
        pin_memory=True,
        drop_last=False,
        persistent_workers=use_persistent_workers,
        collate_fn=collate_fn  # 커스텀 collate 함수 사용
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("🧪 ThreeTaskDataset 테스트")
    
    train_loader, val_loader = create_dataset(batch_size=2, max_samples=100)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    for batch in train_loader:
        print(f"Batch shape: {batch['images'].shape}")
        print(f"Tasks: {batch['tasks']}")
        break 