#!/usr/bin/env python3
"""
ThreeTaskDSPNet Testing Script

Detection + Surface + Depth 3태스크 모델의 성능을 테스트하고 결과를 시각화합니다.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 새로운 구조에 맞는 import
from src.model import ThreeTaskDSPNet, load_pretrained_model
from utils.dataset import create_dataset, DETECTION_CLASSES, SURFACE_LABELS
from configs.config import Config


class ModelTester:
    """
    모델 테스트 및 시각화 클래스.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize model tester.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(checkpoint_path)
        
        # Surface 클래스 색상 매핑
        self.surface_colors = {
            0: [0, 0, 0],       # background - black
            1: [128, 64, 128],  # roadway - purple
            2: [244, 35, 232],  # sidewalk - pink  
            3: [70, 70, 70],    # other surface - gray
        }
        
        print(f"✅ ModelTester 초기화 완료")
        print(f"   모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}개")
        print(f"   디바이스: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device."""
        if device == 'auto':
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and torch.cuda.is_available():
            print(f"🚀 GPU 사용: {torch.cuda.get_device_name()}")
        
        return device
    
    def _load_model(self, checkpoint_path: str) -> ThreeTaskDSPNet:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        print(f"📂 모델 로딩: {checkpoint_path}")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model with same config as training
        if 'config' in checkpoint:
            config = checkpoint['config']
            model = ThreeTaskDSPNet(
                num_detection_classes=config.get('num_detection_classes', 29),
                num_surface_classes=config.get('num_surface_classes', 7),
                input_size=config.get('input_size', (512, 512)),
                pretrained_backbone=config.get('pretrained_backbone', True)
            )
        else:
            # Default config
            model = ThreeTaskDSPNet()
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ 모델 로드 완료")
        return model
    
    def test_single_image(self, image_path: str, save_path: str = None) -> dict:
        """
        Test model on a single image.
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization
            
        Returns:
            Dictionary with predictions and metadata
        """
        # 이미지 로드 및 전처리
        image = self._preprocess_image(image_path)
        
        # 추론
        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0))
        
        # 결과 처리
        results = self._process_outputs(outputs, image_path)
        
        # 시각화
        if save_path:
            self._visualize_results(image_path, results, save_path)
        
        return results
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference."""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512))
        
        # ImageNet 정규화
        image_array = np.array(image) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Tensor로 변환
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        return image_tensor.to(self.device)
    
    def _process_outputs(self, outputs: dict, image_path: str) -> dict:
        """Process model outputs."""
        results = {'image_path': image_path}
        
        # Detection 결과 처리 - 새로운 SSD 표준 형태
        detection_cls = outputs['detection_cls']  # (1, num_anchors, num_classes+1)
        detection_reg = outputs['detection_reg']  # (1, num_anchors, 4)
        
        # Softmax for classification scores (background 포함)
        detection_scores = F.softmax(detection_cls[0], dim=-1)  # (num_anchors, num_classes+1)
        
        # 배경이 아닌 클래스의 최대값 선택
        max_scores, predicted_classes = torch.max(detection_scores[:, 1:], dim=-1)  # background 제외
        predicted_classes += 1  # background 제외했으므로 +1
        
        # 임계값 이상의 예측만 선택
        threshold = 0.5
        valid_indices = max_scores > threshold
        
        if valid_indices.sum() > 0:
            valid_scores = max_scores[valid_indices]
            valid_classes = predicted_classes[valid_indices]
            valid_boxes = detection_reg[0, valid_indices, :4]  # bbox coordinates
            
            results['detections'] = {
                'scores': valid_scores.cpu().numpy(),
                'classes': valid_classes.cpu().numpy(), 
                'boxes': valid_boxes.cpu().numpy(),
                'count': len(valid_scores)
            }
        else:
            results['detections'] = {
                'scores': np.array([]),
                'classes': np.array([]),
                'boxes': np.array([]).reshape(0, 4),
                'count': 0
            }
        
        # Surface 결과 처리
        surface_logits = outputs['surface_segmentation']  # (1, 4, 512, 512)
        surface_pred = torch.argmax(surface_logits[0], dim=0)  # (512, 512)
        
        results['surface_segmentation'] = surface_pred.cpu().numpy()
        
        return results
    
    def _visualize_results(self, image_path: str, results: dict, save_path: str):
        """Visualize and save results."""
        # 원본 이미지 로드
        original_image = Image.open(image_path).convert('RGB')
        original_image = original_image.resize((512, 512))
        original_array = np.array(original_image)
        
        # Figure 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'TwoTaskDSPNet Results: {os.path.basename(image_path)}', fontsize=16)
        
        # 1. 원본 이미지
        axes[0, 0].imshow(original_array)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. Detection 결과
        detection_img = original_array.copy()
        detections = results['detections']
        
        if detections['count'] > 0:
            for i in range(detections['count']):
                score = detections['scores'][i]
                class_id = detections['classes'][i] - 1  # 0-based indexing
                box = detections['boxes'][i]
                
                if 0 <= class_id < len(DETECTION_CLASSES):
                    class_name = DETECTION_CLASSES[class_id]
                    
                    # 박스 그리기 (간단한 방법, 실제로는 좌표 변환 필요)
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1*512), int(y1*512), int(x2*512), int(y2*512)
                    
                    cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 라벨 텍스트
                    label = f'{class_name}: {score:.2f}'
                    cv2.putText(detection_img, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        axes[0, 1].imshow(detection_img)
        axes[0, 1].set_title(f'Object Detection ({detections["count"]} objects)')
        axes[0, 1].axis('off')
        
        # 3. Surface Segmentation 결과
        surface_mask = results['surface_segmentation']
        surface_colored = np.zeros((512, 512, 3), dtype=np.uint8)
        
        for class_id, color in self.surface_colors.items():
            mask = surface_mask == class_id
            surface_colored[mask] = color
        
        axes[1, 0].imshow(surface_colored)
        axes[1, 0].set_title('Surface Segmentation')
        axes[1, 0].axis('off')
        
        # 4. 오버레이 결과
        overlay = original_array.copy().astype(np.float32)
        surface_colored_norm = surface_colored.astype(np.float32) / 255.0
        
        # 배경이 아닌 부분만 오버레이
        non_background = surface_mask > 0
        overlay[non_background] = 0.6 * overlay[non_background] + 0.4 * (surface_colored_norm[non_background] * 255)
        overlay = overlay.astype(np.uint8)
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Original + Surface Overlay')
        axes[1, 1].axis('off')
        
        # 저장
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 시각화 결과 저장: {save_path}")
    
    def test_dataset(self, data_loader, num_samples: int = 10, save_dir: str = "test_results"):
        """Test model on dataset samples."""
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
        
        print(f"\n🧪 데이터셋 테스트 시작 ({num_samples}개 샘플)")
        
        sample_count = 0
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Testing")):
            if sample_count >= num_samples:
                break
            
            images = batch['images'].to(self.device)
            batch_size = images.size(0)
            
            with torch.no_grad():
                outputs = self.model(images)
            
            # 각 이미지에 대해 처리
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                
                # 단일 이미지 결과 추출
                single_outputs = {
                    'detection_cls': outputs['detection_cls'][i:i+1],
                    'detection_reg': outputs['detection_reg'][i:i+1],
                    'surface_segmentation': outputs['surface_segmentation'][i:i+1]
                }
                
                # 원본 이미지 복원 (역정규화)
                image_tensor = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_denorm = image_tensor * std + mean
                image_denorm = torch.clamp(image_denorm, 0, 1)
                
                # PIL Image로 변환
                image_array = (image_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                temp_image_path = f"temp_test_image_{sample_count}.jpg"
                Image.fromarray(image_array).save(temp_image_path)
                
                # 결과 처리 및 시각화
                results = self._process_outputs(single_outputs, temp_image_path)
                save_path = os.path.join(save_dir, f"test_result_{sample_count:03d}.jpg")
                self._visualize_results(temp_image_path, results, save_path)
                
                # 임시 파일 제거
                os.remove(temp_image_path)
                
                sample_count += 1
        
        print(f"✅ 테스트 완료! 결과는 {save_dir}에 저장되었습니다.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='TwoTaskDSPNet Model Testing')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', '-i', type=str,
                       help='Path to single image for testing')
    parser.add_argument('--dataset', '-d', action='store_true',
                       help='Test on dataset samples')
    parser.add_argument('--num_samples', '-n', type=int, default=10,
                       help='Number of samples to test from dataset')
    parser.add_argument('--save_dir', '-s', type=str, default='test_results',
                       help='Directory to save test results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    print("🧪 TwoTaskDSPNet 모델 테스트")
    print("=" * 50)
    
    # 모델 테스터 생성
    tester = ModelTester(args.checkpoint, args.device)
    
    if args.image:
        # 단일 이미지 테스트
        print(f"\n🖼️  단일 이미지 테스트: {args.image}")
        save_path = os.path.join(args.save_dir, f"single_test_{os.path.basename(args.image)}.jpg")
        os.makedirs(args.save_dir, exist_ok=True)
        
        results = tester.test_single_image(args.image, save_path)
        
        print(f"\n📊 테스트 결과:")
        print(f"   Detection: {results['detections']['count']}개 객체 감지")
        print(f"   Surface: 4개 클래스 분할 완료")
        
    elif args.dataset:
        # 데이터셋 테스트
        print(f"\n📚 데이터셋 테스트 ({args.num_samples}개 샘플)")
        
        # 데이터 로더 생성
        _, val_loader = create_dataset(
            batch_size=4,
            num_workers=2,
            max_samples=args.num_samples * 2,
            base_dir="data/original_dataset"
        )
        
        tester.test_dataset(val_loader, args.num_samples, args.save_dir)
    
    else:
        print("❌ --image 또는 --dataset 옵션 중 하나를 선택해주세요.")
        parser.print_help()


if __name__ == "__main__":
    main() 