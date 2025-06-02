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
from typing import Dict

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 새로운 구조에 맞는 import
from src.core.model import ThreeTaskDSPNet
from src.losses.multitask import UberNetMTPSLLoss
from src.data.loaders import create_massive_dataset, create_ubernet_mtpsl_dataset
from configs.config import Config, get_quick_test_config, get_full_training_config, get_massive_dataset_config, get_subset_training_config


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
        
        # 손실함수 - UberNet + MTPSL 방식 사용 (수정됨)
        print("📝 UberNet + MTPSL Loss 사용 (partial label + cross-task consistency)")
        if hasattr(self.config, 'loss'):
            # massive_dataset_config 사용시
            detection_weight = getattr(self.config.loss, 'detection_weight', 1.0)
            surface_weight = getattr(self.config.loss, 'surface_weight', 1.0)
            depth_weight = getattr(self.config.loss, 'depth_weight', 0.5)
        else:
            # 기본 Config 클래스 사용시
            detection_weight = self.config.training.detection_weight
            surface_weight = self.config.training.surface_weight
            depth_weight = self.config.training.depth_weight
            
        # UberNet + MTPSL 손실함수 사용 (partial label handling + cross-task consistency)
        from src.losses.multitask import UberNetMTPSLLoss
        self.criterion = UberNetMTPSLLoss(
            detection_weight=detection_weight,
            surface_weight=surface_weight,
            depth_weight=depth_weight,
            cross_task_weight=0.1,  # Cross-task consistency 가중치
            regularization_weight=0.1,
            num_classes=num_detection_classes if 'num_detection_classes' in locals() else 29
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
    
    def _create_model(self) -> ThreeTaskDSPNet:
        """Create and initialize the model."""
        print(f"📦 모델 생성 중...")
        
        # config 구조에 따라 다르게 접근
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'num_detection_classes'):
            # 기본 Config 클래스 사용시
            num_detection_classes = self.config.model.num_detection_classes
            num_surface_classes = self.config.model.num_surface_classes
            input_size = self.config.model.input_size
            pretrained_backbone = self.config.model.pretrained_backbone
        else:
            # get_massive_dataset_config의 SimpleNamespace 사용시
            num_detection_classes = getattr(self.config.model, 'num_classes', 29)
            num_surface_classes = getattr(self.config.model, 'surface_classes', 7)  
            input_size = getattr(self.config.model, 'input_size', (512, 512))
            pretrained_backbone = getattr(self.config.model, 'pretrained', True)
        
        model = ThreeTaskDSPNet(
            num_detection_classes=num_detection_classes,
            num_surface_classes=num_surface_classes,
            input_size=input_size,
            pretrained_backbone=pretrained_backbone
        )
        
        model = model.to(self.device)
        
        return model
    
    def _create_dataloaders(self):
        """Create data loaders based on configuration."""
        # UberNet + MTPSL 스타일 데이터셋 사용 (수정됨)
        if hasattr(self.config, 'dataset'):
            # 새로운 massive_dataset_config 사용시 - UberNet + MTPSL 방식
            return create_ubernet_mtpsl_dataset(
                base_dir=self.config.dataset.base_dir,
                batch_size=self.config.dataset.batch_size,
                num_workers=self.config.dataset.num_workers,
                max_samples=self.config.dataset.max_samples
            )
        else:
            # 기본 Config 클래스 사용시 - UberNet + MTPSL 방식
            return create_ubernet_mtpsl_dataset(
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
        """Train one epoch with UberNet-style selective updates."""
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
            task_mask = targets['task_mask']
            
            # UberNet 방식: 필요한 태스크만 forward pass
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images, targets, task_mask)  # task_mask 전달
                    loss_dict = self.criterion(outputs, targets, task_mask)
                    loss = loss_dict['total']
            else:
                outputs = self.model(images, targets, task_mask)  # task_mask 전달
                loss_dict = self.criterion(outputs, targets, task_mask)
                loss = loss_dict['total']
            
            # UberNet 방식: Selective gradient update
            self.optimizer.zero_grad()
            
            if self.scaler:
                # Scaled backward for mixed precision
                self.scaler.scale(loss).backward()
                
                # UberNet 방식: 해당 태스크 브랜치 + 백본만 gradient 업데이트
                self._selective_gradient_update(task_mask)
                
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping (업데이트될 파라미터에 대해서만)
                max_grad_norm = getattr(self.config.training, 'max_grad_norm', 1.0)
                self._selective_gradient_clipping(task_mask, max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # UberNet 방식: 해당 태스크 브랜치 + 백본만 gradient 업데이트
                self._selective_gradient_update(task_mask)
                
                # Gradient clipping (업데이트될 파라미터에 대해서만)
                max_grad_norm = getattr(self.config.training, 'max_grad_norm', 1.0)
                self._selective_gradient_clipping(task_mask, max_grad_norm)
                
                self.optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            
            # 태스크별 손실 업데이트 (UberNet 방식)
            if task_mask['detection']:
                task_counts['detection'] += 1
                if 'detection_loss' in loss_dict:
                    det_loss = loss_dict['detection_loss']
                    if torch.is_tensor(det_loss):
                        det_loss = det_loss.item()
                    task_losses['detection'] += det_loss
                    
            if task_mask['surface']:
                task_counts['surface'] += 1
                if 'surface_loss' in loss_dict:
                    surf_loss = loss_dict['surface_loss']
                    if torch.is_tensor(surf_loss):
                        surf_loss = surf_loss.item()
                    task_losses['surface'] += surf_loss
                    
            if task_mask['depth']:
                task_counts['depth'] += 1
                if 'depth_loss' in loss_dict:
                    depth_loss = loss_dict['depth_loss']
                    if torch.is_tensor(depth_loss):
                        depth_loss = depth_loss.item()
                    task_losses['depth'] += depth_loss
            
            # 진행률 표시 업데이트
            active_tasks = [task for task, active in task_mask.items() if active]
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Tasks': '+'.join(active_tasks),
                'Det': f"{task_losses['detection']/(task_counts['detection']+1e-8):.4f}",
                'Surf': f"{task_losses['surface']/(task_counts['surface']+1e-8):.4f}",
                'Depth': f"{task_losses['depth']/(task_counts['depth']+1e-8):.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # 텐서보드 로깅
            if self.writer and batch_idx % self.config.system.print_frequency == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], global_step)
                
                # 태스크별 손실 로깅
                for task_name, task_active in task_mask.items():
                    if task_active and f'{task_name}_loss' in loss_dict:
                        self.writer.add_scalar(f'Train/{task_name.capitalize()}Loss', 
                                             loss_dict[f'{task_name}_loss'], global_step)
        
        # 에폭 통계 계산
        avg_loss = total_loss / len(self.train_loader)
        
        return {
            'avg_loss': avg_loss,
            'detection_loss': task_losses['detection'] / (task_counts['detection'] + 1e-8),
            'surface_loss': task_losses['surface'] / (task_counts['surface'] + 1e-8),
            'depth_loss': task_losses['depth'] / (task_counts['depth'] + 1e-8),
            'task_counts': task_counts
        }
    
    def _selective_gradient_update(self, task_mask: Dict[str, bool]):
        """
        Modified UberNet + Cross-task Consistency 방식:
        - 해당 태스크의 브랜치 + 공유 백본 업데이트 (UberNet)
        - Cross-task consistency를 위해 관련 태스크들도 모두 업데이트
        
        Args:
            task_mask: Which tasks are active in current batch
        """
        # 항상 업데이트할 파라미터: 공유 백본
        backbone_params = set(self.model.backbone.parameters())
        
        # 태스크별 파라미터 수집
        active_task_params = set()
        
        # Detection은 독립적 (라벨이 있을 때만)
        if task_mask.get('detection', False):
            active_task_params.update(self.model.detection_head.parameters())
        
        # 🌉 Cross-task Consistency: surface 또는 depth 중 하나라도 있으면 둘 다 업데이트!
        has_surface_or_depth = (task_mask.get('surface', False) or task_mask.get('depth', False))
        
        if has_surface_or_depth:
            # Surface와 Depth 모두 업데이트 (consistency를 위해)
            active_task_params.update(self.model.surface_head.parameters())
            active_task_params.update(self.model.depth_head.parameters())
            
            # Cross-task projections도 업데이트
            if hasattr(self.model, 'cross_task_heads'):
                active_task_params.update(self.model.cross_task_heads.parameters())
        
        # 업데이트할 파라미터 집합
        params_to_update = backbone_params | active_task_params
        
        # 나머지 파라미터의 gradient를 제거 (Modified UberNet 방식)
        for param in self.model.parameters():
            if param not in params_to_update:
                param.grad = None
    
    def _selective_gradient_clipping(self, task_mask: Dict[str, bool], max_grad_norm: float):
        """
        Modified UberNet + Cross-task Consistency 방식: 업데이트될 파라미터에 대해서만 gradient clipping 적용.
        
        Args:
            task_mask: Which tasks are active in current batch
            max_grad_norm: Maximum gradient norm
        """
        # 업데이트할 파라미터 수집
        params_to_clip = list(self.model.backbone.parameters())
        
        # Detection은 독립적 (라벨이 있을 때만)
        if task_mask.get('detection', False):
            params_to_clip.extend(self.model.detection_head.parameters())
        
        # Cross-task Consistency: surface 또는 depth 중 하나라도 있으면 둘 다 클리핑
        has_surface_or_depth = (task_mask.get('surface', False) or task_mask.get('depth', False))
        
        if has_surface_or_depth:
            params_to_clip.extend(self.model.surface_head.parameters())
            params_to_clip.extend(self.model.depth_head.parameters())
            
            # Cross-task consistency parameters
            if hasattr(self.model, 'cross_task_heads'):
                params_to_clip.extend(self.model.cross_task_heads.parameters())
        
        # Gradient clipping (업데이트될 파라미터에 대해서만)
        if params_to_clip:
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
    
    def _prepare_targets(self, batch: dict) -> dict:
        """Prepare targets for loss computation with UberNet + MTPSL support."""
        targets = {}
        
        # UberNet 방식: Task mask 생성 (배치 레벨)
        task_mask = {
            'detection': False,
            'surface': False, 
            'depth': False
        }
        
        # 배치에서 각 태스크가 존재하는지 확인
        if 'detection_boxes' in batch and len(batch['detection_boxes']) > 0:
            task_mask['detection'] = True
            
        if 'surface_masks' in batch:
            task_mask['surface'] = True
            
        if 'depth_maps' in batch:
            task_mask['depth'] = True
            
        targets['task_mask'] = task_mask
        
        # **Detection targets** - UberNet 방식으로 수정
        if task_mask['detection'] and 'detection_boxes' in batch:
            detection_boxes_list = batch['detection_boxes']
            
            # 각 배치 아이템의 boxes를 device로 이동하고 4차원 bbox만 추출
            processed_boxes = []
            for boxes in detection_boxes_list:
                if torch.is_tensor(boxes):
                    # 5차원에서 4차원 bbox만 추출: [x1, y1, x2, y2, class_id] -> [x1, y1, x2, y2]
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
            detection_labels_list = []
            for boxes in batch['detection_boxes']:
                if torch.is_tensor(boxes) and boxes.size(-1) >= 5:
                    # boxes 형태: [x1, y1, x2, y2, class_id] - 5차원 데이터
                    labels = boxes[:, 4].long()  # 클래스 ID 추출
                    detection_labels_list.append(labels.to(
                        self.device, non_blocking=self.config.system.non_blocking
                    ))
                else:
                    # 빈 텐서 생성
                    detection_labels_list.append(torch.zeros(0, dtype=torch.long).to(self.device))
            
            targets['detection_labels'] = detection_labels_list
        
        # Surface targets - UberNet 방식
        if task_mask['surface'] and 'surface_masks' in batch:
            targets['surface_masks'] = batch['surface_masks'].to(
                self.device, non_blocking=self.config.system.non_blocking
            )
        
        # Depth targets - UberNet 방식
        if task_mask['depth'] and 'depth_maps' in batch:
            targets['depth_maps'] = batch['depth_maps'].to(
                self.device, non_blocking=self.config.system.non_blocking
            )
        
        return targets
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate one epoch with UberNet-style selective computation."""
        self.model.eval()
        total_loss = 0.0
        task_losses = {'detection': 0.0, 'surface': 0.0, 'depth': 0.0}
        task_counts = {'detection': 0, 'surface': 0, 'depth': 0}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="🔍 Validation")
            
            for batch in pbar:
                images = batch['images'].to(self.device, non_blocking=self.config.system.non_blocking)
                targets = self._prepare_targets(batch)
                task_mask = targets['task_mask']
                
                # UberNet 방식: 필요한 태스크만 forward pass
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images, targets, task_mask)  # task_mask 전달
                        loss_dict = self.criterion(outputs, targets, task_mask)
                        loss = loss_dict['total']
                else:
                    outputs = self.model(images, targets, task_mask)  # task_mask 전달
                    loss_dict = self.criterion(outputs, targets, task_mask)
                    loss = loss_dict['total']
                
                total_loss += loss.item()
                
                # 태스크별 손실 업데이트 (UberNet 방식)
                if task_mask['detection'] and 'detection_loss' in loss_dict:
                    task_counts['detection'] += 1
                    det_loss = loss_dict['detection_loss']
                    if torch.is_tensor(det_loss):
                        det_loss = det_loss.item()
                    task_losses['detection'] += det_loss
                    
                if task_mask['surface'] and 'surface_loss' in loss_dict:
                    task_counts['surface'] += 1
                    surf_loss = loss_dict['surface_loss']
                    if torch.is_tensor(surf_loss):
                        surf_loss = surf_loss.item()
                    task_losses['surface'] += surf_loss
                    
                if task_mask['depth'] and 'depth_loss' in loss_dict:
                    task_counts['depth'] += 1
                    depth_loss = loss_dict['depth_loss']
                    if torch.is_tensor(depth_loss):
                        depth_loss = depth_loss.item()
                    task_losses['depth'] += depth_loss
                
                # 활성 태스크 표시
                active_tasks = [task for task, active in task_mask.items() if active]
                pbar.set_postfix({
                    'Val Loss': f"{loss.item():.4f}",
                    'Tasks': '+'.join(active_tasks)
                })
        
        avg_val_loss = total_loss / len(self.val_loader)
        return {
            'avg_loss': avg_val_loss,
            'detection_loss': task_losses['detection'] / (task_counts['detection'] + 1e-8),
            'surface_loss': task_losses['surface'] / (task_counts['surface'] + 1e-8),
            'depth_loss': task_losses['depth'] / (task_counts['depth'] + 1e-8),
            'task_counts': task_counts
        }
    
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
    
    # 설정 로드 - 부분 데이터셋으로 테스트
    config = get_subset_training_config()  # ✅ 부분 데이터셋으로 빠른 테스트
    # config = get_massive_dataset_config()  # 전체 데이터셋 설정 (나중에 사용)
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