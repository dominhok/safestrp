#!/usr/bin/env python3
"""
SafeStrp Training Script with YAML Config

깔끔한 YAML 기반 설정 시스템을 사용한 훈련 스크립트.
핵심 설정만 YAML로 컨트롤하고 나머지는 합리적 기본값 사용.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 리팩토링된 구조에 맞는 YAML config 시스템
from configs.yaml_config import load_training_config, save_config, quick_config

# 리팩토링된 구조에 맞는 import
from src.core.model import ThreeTaskDSPNet
from src.losses.multitask import SimpleTwoTaskLoss
from src.data.loaders import create_dataset


class YAMLTrainer:
    """YAML config 기반 깔끔한 훈련 클래스."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize trainer with YAML config.
        
        Args:
            config_path: YAML config 파일 경로 (None이면 자동 탐색)
        """
        self.config = load_training_config(config_path)
        
        # 디바이스 설정
        self.device = self._setup_device()
        
        # 모델 생성
        self.model = self._create_model()
        
        # 손실함수
        self.criterion = self._create_loss_function()
        
        # 옵티마이저
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 데이터로더
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Mixed precision scaler
        self.scaler = self._create_scaler()
        
        # 로깅
        self.writer = self._create_tensorboard_writer()
        
        # 체크포인트 관리
        self.best_loss = float('inf')
        self.current_epoch = 0
        
        print("✅ YAML Trainer 초기화 완료")
        self._print_config_summary()
    
    def _setup_device(self) -> torch.device:
        """디바이스 설정."""
        device_setting = self.config.system.device
        
        if device_setting == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"🚀 GPU 사용: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                print("💻 CPU 사용")
        else:
            device = torch.device(device_setting)
            print(f"🎯 사용자 지정 디바이스: {device}")
        
        return device
    
    def _create_model(self) -> ThreeTaskDSPNet:
        """모델 생성."""
        model = ThreeTaskDSPNet(
            num_detection_classes=self.config.model.num_detection_classes,
            num_surface_classes=self.config.model.num_surface_classes,
            input_size=tuple(self.config.model.input_size),
            pretrained_backbone=self.config.model.pretrained_backbone
        )
        return model.to(self.device)
    
    def _create_loss_function(self) -> SimpleTwoTaskLoss:
        """손실함수 생성."""
        return SimpleTwoTaskLoss(
            detection_weight=self.config.training.loss_weights.detection,
            surface_weight=self.config.training.loss_weights.surface,
            distance_weight=self.config.training.loss_weights.depth
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """옵티마이저 생성."""
        optimizer_name = self.config.training.optimizer.lower()
        
        if optimizer_name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")
    
    def _create_scheduler(self):
        """스케줄러 생성."""
        scheduler_name = self.config.training.scheduler.lower()
        
        if scheduler_name == "reducelronplateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.training.patience,
                verbose=True
            )
        elif scheduler_name == "cosineannealingwarmrestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2
            )
        elif scheduler_name == "none":
            return None
        else:
            print(f"⚠️  알 수 없는 스케줄러: {scheduler_name}. 스케줄러를 사용하지 않습니다.")
            return None
    
    def _create_dataloaders(self):
        """데이터로더 생성."""
        return create_dataset(
            base_dir=self.config.dataset.base_dir,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.system.num_workers,
            max_samples=self.config.dataset.max_samples
        )
    
    def _create_scaler(self):
        """Mixed precision scaler 생성."""
        if self.config.training.mixed_precision and self.device.type == 'cuda':
            return torch.amp.GradScaler('cuda')
        return None
    
    def _create_tensorboard_writer(self):
        """TensorBoard writer 생성."""
        if self.config.logging.tensorboard:
            log_dir = os.path.join(self.config.logging.log_dir, "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            return SummaryWriter(log_dir)
        return None
    
    def _print_config_summary(self):
        """설정 요약 출력."""
        print("\n" + "="*50)
        print("📋 훈련 설정 요약")
        print("="*50)
        print(f"🏗️  모델: Detection({self.config.model.num_detection_classes}) + Surface({self.config.model.num_surface_classes}) + Depth")
        print(f"📊 배치 크기: {self.config.training.batch_size}")
        print(f"🎯 학습률: {self.config.training.learning_rate}")
        print(f"🏃 에포크: {self.config.training.epochs}")
        print(f"⚖️  손실 가중치: Det={self.config.training.loss_weights.detection}, Surf={self.config.training.loss_weights.surface}, Depth={self.config.training.loss_weights.depth}")
        print(f"🔧 옵티마이저: {self.config.training.optimizer}")
        print(f"📈 스케줄러: {self.config.training.scheduler}")
        print(f"⚡ Mixed Precision: {self.config.training.mixed_precision}")
        print(f"💾 체크포인트: {self.config.checkpoint.save_dir}")
        print("="*50 + "\n")
    
    def _prepare_targets(self, batch):
        """배치에서 타겟 준비."""
        targets = {}
        
        # Surface segmentation targets
        if 'surface_masks' in batch:
            targets['surface_masks'] = batch['surface_masks'].to(
                self.device, non_blocking=self.config.system.non_blocking
            )
        
        # Detection targets (SSD 형태)
        if 'detection_boxes' in batch:
            detection_boxes_list = []
            for boxes in batch['detection_boxes']:
                if torch.is_tensor(boxes) and boxes.size(-1) >= 4:
                    bbox_only = boxes[:, :4]  # bbox 좌표만 추출
                    detection_boxes_list.append(bbox_only.to(
                        self.device, non_blocking=self.config.system.non_blocking
                    ))
                else:
                    detection_boxes_list.append(torch.zeros(0, 4, dtype=torch.float32).to(self.device))
            
            targets['detection_boxes'] = detection_boxes_list
        
        # Detection labels
        if 'detection_labels' in batch:
            detection_labels_list = []
            for labels in batch['detection_labels']:
                if torch.is_tensor(labels):
                    detection_labels_list.append(labels.to(
                        self.device, non_blocking=self.config.system.non_blocking
                    ))
                else:
                    detection_labels_list.append(torch.zeros(0, dtype=torch.long).to(self.device))
            
            targets['detection_labels'] = detection_labels_list
        
        # Depth targets
        if 'depth_maps' in batch:
            targets['depth_maps'] = batch['depth_maps'].to(
                self.device, non_blocking=self.config.system.non_blocking
            )
            
        if 'confidence_masks' in batch:
            targets['confidence_masks'] = batch['confidence_masks'].to(
                self.device, non_blocking=self.config.system.non_blocking
            )
        
        return targets
    
    def train_epoch(self, epoch: int) -> dict:
        """한 에포크 훈련."""
        self.model.train()
        total_loss = 0.0
        task_losses = {'detection': 0.0, 'surface': 0.0, 'depth': 0.0}
        task_counts = {'detection': 0, 'surface': 0, 'depth': 0}
        
        pbar = tqdm(self.train_loader, desc=f"🎯 Epoch {epoch+1}/{self.config.training.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # 데이터를 디바이스로 이동
            images = batch['images'].to(self.device, non_blocking=self.config.system.non_blocking)
            
            # 타겟 준비
            targets = self._prepare_targets(batch)
            
            # Forward pass
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss, loss_details = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss, loss_details = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
            
            # 손실 누적
            total_loss += loss.item()
            for task, task_loss in loss_details.items():
                if task in task_losses:
                    task_losses[task] += task_loss
                    task_counts[task] += 1
            
            # 진행 상황 업데이트
            if batch_idx % self.config.logging.print_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Det': f'{task_losses["detection"]/(task_counts["detection"]+1e-6):.4f}',
                    'Surf': f'{task_losses["surface"]/(task_counts["surface"]+1e-6):.4f}',
                    'Depth': f'{task_losses["depth"]/(task_counts["depth"]+1e-6):.4f}'
                })
        
        # 평균 손실 계산
        avg_loss = total_loss / len(self.train_loader)
        avg_task_losses = {task: loss / max(count, 1) for task, loss, count in 
                          zip(task_losses.keys(), task_losses.values(), task_counts.values())}
        
        return {
            'total_loss': avg_loss,
            'task_losses': avg_task_losses
        }
    
    def validate(self) -> dict:
        """검증."""
        self.model.eval()
        total_loss = 0.0
        task_losses = {'detection': 0.0, 'surface': 0.0, 'depth': 0.0}
        task_counts = {'detection': 0, 'surface': 0, 'depth': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="🔍 Validation"):
                images = batch['images'].to(self.device, non_blocking=self.config.system.non_blocking)
                targets = self._prepare_targets(batch)
                
                outputs = self.model(images)
                loss, loss_details = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                for task, task_loss in loss_details.items():
                    if task in task_losses:
                        task_losses[task] += task_loss
                        task_counts[task] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        avg_task_losses = {task: loss / max(count, 1) for task, loss, count in 
                          zip(task_losses.keys(), task_losses.values(), task_counts.values())}
        
        return {
            'total_loss': avg_loss,
            'task_losses': avg_task_losses
        }
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """체크포인트 저장."""
        os.makedirs(self.config.checkpoint.save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'config': self.config.to_dict()
        }
        
        # 일반 체크포인트 저장
        if epoch % self.config.checkpoint.save_interval == 0:
            checkpoint_path = os.path.join(self.config.checkpoint.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 체크포인트 저장: {checkpoint_path}")
        
        # 최고 성능 체크포인트 저장
        if is_best and self.config.checkpoint.keep_best:
            best_path = os.path.join(self.config.checkpoint.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"🏆 최고 성능 모델 저장: {best_path}")
    
    def train(self):
        """전체 훈련 프로세스."""
        print("🚀 훈련 시작!")
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # 훈련
            train_results = self.train_epoch(epoch)
            
            # 검증
            val_results = self.validate()
            
            # 스케줄러 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['total_loss'])
                else:
                    self.scheduler.step()
            
            # 로깅
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n📊 Epoch {epoch+1}/{self.config.training.epochs}")
            print(f"   훈련 손실: {train_results['total_loss']:.4f}")
            print(f"   검증 손실: {val_results['total_loss']:.4f}")
            print(f"   학습률: {current_lr:.2e}")
            
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_results['total_loss'], epoch)
                self.writer.add_scalar('Loss/Val', val_results['total_loss'], epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                for task, loss in train_results['task_losses'].items():
                    self.writer.add_scalar(f'Loss_Train/{task}', loss, epoch)
                for task, loss in val_results['task_losses'].items():
                    self.writer.add_scalar(f'Loss_Val/{task}', loss, epoch)
            
            # 체크포인트 저장
            is_best = val_results['total_loss'] < self.best_loss
            if is_best:
                self.best_loss = val_results['total_loss']
            
            self.save_checkpoint(epoch, val_results['total_loss'], is_best)
        
        print("🎉 훈련 완료!")
        if self.writer:
            self.writer.close()


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(description='SafeStrp YAML Training')
    parser.add_argument('--config', type=str, default=None, 
                       help='YAML config 파일 경로 (기본값: 자동 탐색)')
    parser.add_argument('--quick', action='store_true',
                       help='빠른 테스트를 위한 축소 설정 사용')
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            # 빠른 테스트용 설정
            config = quick_config(
                batch_size=4,
                epochs=5,
                max_samples=100
            )
            print("🏃 빠른 테스트 모드")
        else:
            config = None  # 자동 로드
        
        trainer = YAMLTrainer(args.config)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n⏹️  훈련이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main() 