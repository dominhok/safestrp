#!/usr/bin/env python3
"""
TwoTaskDSPNet Training Script

Detection + Surface 2태스크 멀티태스크 학습을 위한 메인 훈련 스크립트.
새로운 프로젝트 구조에 맞게 최적화되었습니다.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from tqdm import tqdm

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 새로운 구조에 맞는 import
from src.model import TwoTaskDSPNet
from src.losses import SimpleTwoTaskLoss, AdvancedTwoTaskLoss
from utils.dataset import create_massive_dataset
from configs.config import Config, get_quick_test_config, get_full_training_config, get_massive_dataset_config


class TwoTaskTrainer:
    """
    2태스크 멀티태스크 트레이너.
    
    새로운 프로젝트 구조와 설정 시스템을 활용합니다.
    """
    
    def __init__(self, config: Config):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration object with all parameters
        """
        self.config = config
        
        # Device 설정
        self.device = self._setup_device()
        
        # 모델 생성
        self.model = self._create_model()
        
        # 데이터 로더 생성
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # 옵티마이저 및 스케줄러
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 손실함수 - 안정성을 위해 Simple 버전 사용
        print("📝 Simple Loss 사용 (안정화 버전)")
        if hasattr(self.config, 'loss'):
            # massive_dataset_config 사용시
            detection_weight = getattr(self.config.loss, 'detection_weight', 1.0)
            surface_weight = getattr(self.config.loss, 'surface_weight', 1.0)
            distance_weight = getattr(self.config.loss, 'distance_weight', 0.5)
        else:
            # 기본 Config 클래스 사용시
            detection_weight = self.config.training.detection_weight
            surface_weight = self.config.training.surface_weight
            distance_weight = self.config.training.distance_weight
            
        self.criterion = SimpleTwoTaskLoss(
            detection_weight=detection_weight,
            surface_weight=surface_weight,
            distance_weight=distance_weight
        )
        
        # 디렉토리 생성 - config 구조에 따라 다르게 접근
        if hasattr(self.config, 'save'):
            checkpoint_dir = self.config.save.checkpoint_dir
            log_dir = getattr(self.config.logging, 'log_dir', 'logs/two_task_training')
        else:
            checkpoint_dir = self.config.data.checkpoint_dir
            log_dir = self.config.data.log_dir
            
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 텐서보드
        use_tensorboard = False
        if hasattr(self.config, 'logging'):
            use_tensorboard = getattr(self.config.logging, 'use_tensorboard', True)
        elif hasattr(self.config, 'system'):
            use_tensorboard = self.config.system.tensorboard_log
            
        self.writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else None
        
        # 훈련 상태
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Mixed precision 설정
        if hasattr(self.config, 'system') and self.config.system.mixed_precision and self.device == "cuda":
            print("⚡ Mixed Precision 훈련 활성화")
            self.scaler = torch.amp.GradScaler('cuda')
        elif hasattr(self.config, 'training') and self.config.training.use_amp and self.device == "cuda":
            print("⚡ Mixed Precision 훈련 활성화")
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        print(f"✅ TwoTaskTrainer 초기화 완료")
        print(f"   모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}개")
        print(f"   훈련 샘플: {len(self.train_loader.dataset)}개")
        print(f"   검증 샘플: {len(self.val_loader.dataset)}개")
        print(f"   디바이스: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup device based on configuration."""
        if self.config.system.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.system.device
        
        if device == "cuda":
            print(f"🚀 GPU 사용: {torch.cuda.get_device_name()}")
            print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        return device
    
    def _create_model(self) -> TwoTaskDSPNet:
        """Create model based on configuration."""
        # config 구조에 따라 다르게 접근
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'num_detection_classes'):
            # 기본 Config 클래스 사용시
            num_detection_classes = self.config.model.num_detection_classes
            num_surface_classes = self.config.model.num_surface_classes
            input_size = self.config.model.input_size
            pretrained_backbone = self.config.model.pretrained_backbone
        else:
            # get_massive_dataset_config의 SimpleNamespace 사용시
            num_detection_classes = getattr(self.config.model, 'num_classes', 27)
            num_surface_classes = getattr(self.config.model, 'surface_classes', 4)  
            input_size = getattr(self.config.model, 'input_size', (512, 512))
            pretrained_backbone = getattr(self.config.model, 'pretrained', True)
        
        model = TwoTaskDSPNet(
            num_detection_classes=num_detection_classes,
            num_surface_classes=num_surface_classes,
            input_size=input_size,
            pretrained_backbone=pretrained_backbone
        )
        
        model = model.to(self.device)
        
        return model
    
    def _create_dataloaders(self):
        """Create data loaders based on configuration."""
        # 새로운 config 구조 체크
        if hasattr(self.config, 'dataset'):
            # 새로운 massive_dataset_config 사용
            return create_massive_dataset(
                base_dir=self.config.dataset.base_dir,
                batch_size=self.config.dataset.batch_size,
                num_workers=self.config.dataset.num_workers,
                max_samples=self.config.dataset.max_samples,
                use_depth=self.config.dataset.use_depth
            )
        else:
            # 기존 Config 클래스 사용
            return create_massive_dataset(
                batch_size=self.config.training.batch_size,
                num_workers=self.config.training.num_workers,
                max_samples=self.config.training.max_samples,
                base_dir=self.config.data.base_dir
            )
    
    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        # config 구조에 따라 다르게 접근
        if hasattr(self.config, 'optimizer'):
            # massive_dataset_config 사용시
            optimizer_type = getattr(self.config.optimizer, 'type', 'adamw').lower()
            learning_rate = getattr(self.config.optimizer, 'lr', 1e-4)
            weight_decay = getattr(self.config.optimizer, 'weight_decay', 1e-4)
        else:
            # 기본 Config 클래스 사용시
            optimizer_type = self.config.training.optimizer_type.lower()
            learning_rate = self.config.training.learning_rate
            weight_decay = self.config.training.weight_decay
            
        if optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on configuration."""
        # config 구조에 따라 다르게 접근
        if hasattr(self.config, 'scheduler'):
            # massive_dataset_config 사용시
            scheduler_type = getattr(self.config.scheduler, 'type', 'cosine').lower()
            epochs = getattr(self.config.training, 'epochs', 50)
        else:
            # 기본 Config 클래스 사용시
            scheduler_type = self.config.training.scheduler_type.lower()
            epochs = self.config.training.epochs
            
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        elif scheduler_type == "step":
            step_size = getattr(self.config.training, 'step_size', 15)
            gamma = getattr(self.config.training, 'gamma', 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type in ["plateau", "reduceonplateau"]:
            factor = getattr(self.config.scheduler, 'factor', 0.5) if hasattr(self.config, 'scheduler') else 0.1
            patience = getattr(self.config.scheduler, 'patience', 5) if hasattr(self.config, 'scheduler') else self.config.training.patience // 2
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=patience,
                factor=factor
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        task_losses = {'detection': 0.0, 'surface': 0.0}
        task_counts = {'detection': 0, 'surface': 0}
        
        pbar = tqdm(self.train_loader, desc=f"🎯 Epoch {epoch+1}/{self.config.training.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # 데이터를 디바이스로 이동
            images = batch['images'].to(self.device, non_blocking=self.config.system.non_blocking)
            
            # 타겟 준비
            targets = self._prepare_targets(batch)
            
            # Mixed precision 사용시
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss, loss_details = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss, loss_details = self.criterion(outputs, targets)
            
            # 역전파
            self.optimizer.zero_grad()
            
            if self.scaler:
                # Scaled backward for mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping - config 구조에 따라 다르게 접근
                max_grad_norm = getattr(self.config.training, 'max_grad_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping - config 구조에 따라 다르게 접근  
                max_grad_norm = getattr(self.config.training, 'max_grad_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                self.optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            
            # 태스크별 손실 업데이트
            tasks = batch['tasks']
            for task in tasks:
                if task == 'detection':
                    task_counts['detection'] += 1
                    if 'cls_loss' in loss_details:
                        det_loss = loss_details['cls_loss']
                        if torch.is_tensor(det_loss):
                            det_loss = det_loss.item()
                        task_losses['detection'] += det_loss
                elif task == 'surface':
                    task_counts['surface'] += 1
                    if 'surface_loss' in loss_details:
                        surf_loss = loss_details['surface_loss']
                        if torch.is_tensor(surf_loss):
                            surf_loss = surf_loss.item()
                        task_losses['surface'] += surf_loss
            
            # 진행률 표시 업데이트
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Det': f"{task_losses['detection']/(task_counts['detection']+1e-8):.4f}",
                'Surf': f"{task_losses['surface']/(task_counts['surface']+1e-8):.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # 텐서보드 로깅
            if self.writer and batch_idx % self.config.system.print_frequency == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], global_step)
        
        # 에폭 통계 계산
        avg_loss = total_loss / len(self.train_loader)
        
        return {
            'avg_loss': avg_loss,
            'detection_loss': task_losses['detection'] / (task_counts['detection'] + 1e-8),
            'surface_loss': task_losses['surface'] / (task_counts['surface'] + 1e-8)
        }
    
    def _prepare_targets(self, batch: dict) -> dict:
        """Prepare targets for loss computation."""
        targets = {
            'has_detection': batch['has_detection'],
            'has_surface': batch['has_surface']
        }
        
        # **Detection targets** (수정됨 - 리스트 형태 처리)
        if 'detection_boxes' in batch:
            # detection_boxes는 리스트 형태로 들어옴 (가변 길이 때문에)
            detection_boxes_list = batch['detection_boxes']
            
            # 각 배치 아이템의 boxes를 device로 이동하고 4차원 bbox만 추출
            processed_boxes = []
            for boxes in detection_boxes_list:
                if torch.is_tensor(boxes):
                    # 6차원에서 4차원 bbox만 추출: [x1, y1, x2, y2, class_id, distance] -> [x1, y1, x2, y2]
                    if boxes.size(-1) >= 4:
                        bbox_only = boxes[:, :4]  # 첫 4개 차원만 사용
                        processed_boxes.append(bbox_only.to(
                            self.device, non_blocking=self.config.system.non_blocking
                        ))
                    else:
                        # 빈 텐서 생성
                        processed_boxes.append(torch.zeros(0, 4, dtype=torch.float32).to(self.device))
                else:
                    # boxes가 리스트인 경우 텐서로 변환하고 4차원만 추출
                    if len(boxes) > 0 and len(boxes[0]) >= 4:
                        bbox_only = [[box[0], box[1], box[2], box[3]] for box in boxes]
                        processed_boxes.append(torch.tensor(
                            bbox_only, dtype=torch.float32
                        ).to(self.device, non_blocking=self.config.system.non_blocking))
                    else:
                        processed_boxes.append(torch.zeros(0, 4, dtype=torch.float32).to(self.device))
            
            targets['detection_boxes'] = processed_boxes
        
        # Detection labels는 boxes에서 추출 (클래스 정보가 포함됨)
        if 'detection_boxes' in batch:
            detection_labels_list = []
            for boxes in batch['detection_boxes']:
                if torch.is_tensor(boxes) and boxes.size(-1) >= 6:
                    # boxes 형태: [x1, y1, x2, y2, class_id, distance]
                    labels = boxes[:, 4].long()  # 클래스 ID 추출
                    detection_labels_list.append(labels.to(
                        self.device, non_blocking=self.config.system.non_blocking
                    ))
                else:
                    # 빈 텐서 생성
                    detection_labels_list.append(torch.zeros(0, dtype=torch.long).to(self.device))
            
            targets['detection_labels'] = detection_labels_list
        
        # Distance targets도 boxes에서 추출
        if 'detection_boxes' in batch:
            detection_distances_list = []
            for boxes in batch['detection_boxes']:
                if torch.is_tensor(boxes) and boxes.size(-1) >= 6:
                    # boxes 형태: [x1, y1, x2, y2, class_id, distance]
                    distances = boxes[:, 5]  # distance 추출
                    detection_distances_list.append(distances.to(
                        self.device, non_blocking=self.config.system.non_blocking
                    ))
                else:
                    # 빈 텐서 생성
                    detection_distances_list.append(torch.zeros(0, dtype=torch.float32).to(self.device))
            
            targets['detection_distances'] = detection_distances_list
        
        # Surface targets
        if 'surface_masks' in batch:
            targets['surface_masks'] = batch['surface_masks'].to(
                self.device, non_blocking=self.config.system.non_blocking
            )
        
        return targets
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="🔍 Validation")
            
            for batch in pbar:
                images = batch['images'].to(self.device, non_blocking=self.config.system.non_blocking)
                targets = self._prepare_targets(batch)
                
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        loss, _ = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss, _ = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                pbar.set_postfix({'Val Loss': f"{loss.item():.4f}"})
        
        avg_val_loss = total_loss / len(self.val_loader)
        return {'avg_loss': avg_val_loss}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        if self.config.system.save_last:
            latest_path = os.path.join(self.config.data.checkpoint_dir, 'latest_two_task.pth')
            torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best and self.config.system.save_best_only:
            best_path = os.path.join(self.config.data.checkpoint_dir, 'best_two_task.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved: {best_path}")
    
    def train(self):
        """Main training loop."""
        print(f"\n🚀 훈련 시작!")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            # 훈련
            train_metrics = self.train_epoch(epoch)
            
            # 검증
            if epoch % self.config.training.val_frequency == 0:
                val_metrics = self.validate_epoch(epoch)
                
                # 최고 성능 모델 체크
                is_best = val_metrics['avg_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['avg_loss']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # 체크포인트 저장
                if epoch % self.config.training.save_frequency == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
                
                # 텐서보드 로깅
                if self.writer:
                    self.writer.add_scalar('Train/EpochLoss', train_metrics['avg_loss'], epoch)
                    self.writer.add_scalar('Val/EpochLoss', val_metrics['avg_loss'], epoch)
                
                # 결과 출력
                print(f"\n📊 Epoch {epoch+1} 결과:")
                print(f"   훈련 손실: {train_metrics['avg_loss']:.4f}")
                print(f"   검증 손실: {val_metrics['avg_loss']:.4f}")
                print(f"   최고 성능: {self.best_val_loss:.4f}")
                print(f"   학습률: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Early stopping 체크
                if self.patience_counter >= self.config.training.patience:
                    print(f"⏹️  Early stopping! (patience: {self.config.training.patience})")
                    break
            
            # 스케줄러 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['avg_loss'])
                else:
                    self.scheduler.step()
        
        # 훈련 완료
        total_time = time.time() - start_time
        print(f"\n✅ 훈련 완료!")
        print(f"   총 시간: {total_time/3600:.2f}시간")
        print(f"   최고 검증 손실: {self.best_val_loss:.4f}")
        
        if self.writer:
            self.writer.close()


def main():
    """Main function."""
    print("🎯 TwoTaskDSPNet 훈련 시작")
    print("=" * 70)
    
    # 설정 로드 - 대규모 데이터셋 설정 사용
    config = get_massive_dataset_config()  # 전체 데이터셋 설정
    # config = get_full_training_config()  # 전체 훈련 설정
    # config = get_quick_test_config()  # 빠른 테스트 설정
    
    # 설정 출력
    config.print_config()
    
    # 데이터셋 정보 출력
    print(f"\n📊 데이터셋 정보:")
    print(f"   기본 경로: {config.data.base_dir}")
    print(f"   최대 샘플 수: {config.dataset.max_samples:,}개")
    print(f"   배치 크기: {config.dataset.batch_size}")
    print(f"   Depth 사용: {config.dataset.use_depth}")
    
    # 재현성 설정
    if config.system.seed:
        torch.manual_seed(config.system.seed)
        np.random.seed(config.system.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.system.seed)
    
    # 트레이너 생성 및 훈련 시작
    trainer = TwoTaskTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 