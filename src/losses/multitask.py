"""
Multi-task loss functions for SafeStrp.

Contains combined losses for detection, segmentation and depth tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .base import FocalLoss
from .task_specific import CrossEntropySegmentationLoss, DepthLoss
from .utils import iou_match_anchors, hard_negative_mining


class SimpleTwoTaskLoss(nn.Module):
    """
    Simplified loss for Detection + Surface tasks (원본 DSPNet 방식).
    
    Detection에 distance가 통합된 2태스크 구조.
    """
    
    def __init__(self, 
                 detection_weight: float = 1.0,
                 surface_weight: float = 1.0,
                 distance_weight: float = 0.5):
        """
        Initialize simple two-task loss (Detection + Surface).
        
        Args:
            detection_weight: Weight for detection loss
            surface_weight: Weight for surface segmentation loss
            distance_weight: Weight for distance regression loss
        """
        super().__init__()
        
        self.detection_weight = detection_weight
        self.surface_weight = surface_weight
        self.distance_weight = distance_weight
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.segmentation_loss = CrossEntropySegmentationLoss(ignore_index=255)
        
    def forward(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate multi-task loss.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        losses = {}
        device = next(iter(outputs.values())).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Detection loss (simplified)
        if 'detection_cls' in outputs:
            cls_preds = outputs['detection_cls']
            reg_preds = outputs['detection_reg']
            
            # Simplified classification loss
            batch_size, num_anchors, num_classes = cls_preds.shape
            fake_targets = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
            
            cls_loss = self.focal_loss(cls_preds.view(-1, num_classes), fake_targets.view(-1)).mean()
            reg_loss = torch.mean(reg_preds ** 2) * 0.01
            
            losses['cls_loss'] = cls_loss
            losses['bbox_loss'] = reg_loss
            total_loss = total_loss + self.detection_weight * (cls_loss + reg_loss)
        
        # Surface segmentation loss
        if 'surface_segmentation' in outputs and 'surface_masks' in targets:
            surface_loss = self.segmentation_loss(outputs['surface_segmentation'], targets['surface_masks'])
            losses['surface_loss'] = surface_loss
            total_loss = total_loss + self.surface_weight * surface_loss
        
        losses['total_loss'] = total_loss
        return total_loss, losses


class UberNetMTPSLLoss(nn.Module):
    """
    UberNet + MTPSL 하이브리드 손실함수.
    
    UberNet의 partial label handling과 MTPSL의 cross-task consistency를 결합.
    """
    
    def __init__(self,
                 detection_weight: float = 1.0,
                 surface_weight: float = 0.5,
                 depth_weight: float = 1.0,
                 cross_task_weight: float = 0.1,
                 regularization_weight: float = 0.1,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 num_classes: int = 29):
        super().__init__()
        
        # Loss weights
        self.detection_weight = detection_weight
        self.surface_weight = surface_weight
        self.depth_weight = depth_weight
        self.cross_task_weight = cross_task_weight
        self.regularization_weight = regularization_weight
        self.num_classes = num_classes
        
        # Individual loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.surface_ce_loss = CrossEntropySegmentationLoss()
        self.depth_l1_loss = DepthLoss(loss_type='l1')

    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                task_mask: Dict[str, bool]) -> Dict[str, torch.Tensor]:
        """
        Compute UberNet + MTPSL combined loss.
        
        Args:
            predictions: Model predictions for all tasks
            targets: Ground truth targets for available tasks
            task_mask: Which tasks have labels in this batch
            
        Returns:
            Dictionary of computed losses
        """
        # Device 추출을 안전하게 수정
        device = None
        for value in predictions.values():
            if torch.is_tensor(value):
                device = value.device
                break
        
        if device is None:
            # Fallback to first tensor in targets
            for value in targets.values():
                if torch.is_tensor(value):
                    device = value.device
                    break
                    
        if device is None:
            device = torch.device('cpu')  # Final fallback
            
        losses = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 1. Detection loss (only if detection labels available)
        if task_mask.get('detection', False) and 'detection_cls' in predictions:
            det_loss, det_reg_loss = self._compute_detection_loss(predictions, targets)
            losses['detection_loss'] = det_loss + det_reg_loss
            total_loss = total_loss + self.detection_weight * (det_loss + det_reg_loss)
        
        # 2. Surface segmentation loss (only if surface labels available)
        if task_mask.get('surface', False) and 'surface_segmentation' in predictions:
            surface_loss = self._compute_surface_loss(predictions, targets)
            losses['surface_loss'] = surface_loss
            total_loss = total_loss + self.surface_weight * surface_loss
        
        # 3. Depth estimation loss (only if depth labels available)
        if task_mask.get('depth', False) and 'depth_estimation' in predictions:
            depth_loss = self._compute_depth_loss(predictions, targets)
            losses['depth_loss'] = depth_loss
            total_loss = total_loss + self.depth_weight * depth_loss
        
        # 4. Cross-task consistency loss
        if self.cross_task_weight > 0 and 'cross_task_embeddings' in predictions:
            cross_task_loss = self._compute_cross_task_consistency_loss(predictions, targets, task_mask)
            losses['cross_task_loss'] = cross_task_loss
            total_loss = total_loss + self.cross_task_weight * cross_task_loss
        
        losses['total'] = total_loss
        return losses

    def _compute_detection_loss(self, predictions: Dict, targets: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute detection loss (simplified)."""
        device = predictions['detection_cls'].device
        
        cls_pred = predictions['detection_cls']  # (B, num_anchors, num_classes)
        reg_pred = predictions['detection_reg']  # (B, num_anchors, 4)
        
        batch_size, num_anchors, num_classes = cls_pred.shape
        
        # Simplified targets
        dummy_labels = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
        cls_loss = self.focal_loss(cls_pred.view(-1, num_classes), dummy_labels.view(-1))
        reg_loss = torch.mean(reg_pred ** 2) * 0.01
        
        return cls_loss, reg_loss

    def _compute_surface_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """Compute surface segmentation loss."""
        surface_pred = predictions['surface_segmentation']  # (B, C, H, W)
        surface_targets = targets['surface_masks']  # (N, H, W) where N <= B
        
        # 배치 크기 확인
        batch_size = surface_pred.shape[0]
        target_size = surface_targets.shape[0] 
        
        if batch_size != target_size:
            # 배치의 task_masks에서 surface가 True인 인덱스 찾기
            surface_indices = self._find_task_indices(targets, 'surface')
            
            # surface가 있는 샘플만 선택
            if len(surface_indices) > 0:
                # 리스트를 tensor로 변환
                surface_indices = torch.tensor(surface_indices, device=surface_pred.device)
                surface_pred = surface_pred[surface_indices]
            else:
                # surface 태스크가 없으면 dummy loss 반환
                return torch.tensor(0.0, device=surface_pred.device, requires_grad=True)
        
        return self.surface_ce_loss(surface_pred, surface_targets)

    def _compute_depth_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """Compute depth estimation loss."""
        depth_pred = predictions['depth_estimation']  # (B, 1, H, W)
        depth_targets = targets['depth_maps']  # (N, H, W) where N <= B
        
        # 배치 크기 확인
        batch_size = depth_pred.shape[0]
        target_size = depth_targets.shape[0]
        
        if batch_size != target_size:
            # 배치의 task_masks에서 depth가 True인 인덱스 찾기
            depth_indices = self._find_task_indices(targets, 'depth')
            
            # depth가 있는 샘플만 선택
            if len(depth_indices) > 0:
                # 리스트를 tensor로 변환
                depth_indices = torch.tensor(depth_indices, device=depth_pred.device)
                depth_pred = depth_pred[depth_indices]
            else:
                # depth 태스크가 없으면 dummy loss 반환
                return torch.tensor(0.0, device=depth_pred.device, requires_grad=True)
        
        return self.depth_l1_loss(depth_pred, depth_targets)

    def _find_task_indices(self, targets: Dict, task_name: str) -> List[int]:
        """
        배치에서 특정 태스크가 있는 샘플들의 인덱스를 찾는다.
        
        Args:
            targets: 타겟 딕셔너리 (task_masks 포함)
            task_name: 찾을 태스크 이름 ('surface', 'depth', 'detection')
            
        Returns:
            해당 태스크가 있는 샘플들의 인덱스 리스트
        """
        task_indices = []
        
        # task_masks 가져오기 - 여러 형태를 지원
        if 'task_masks' in targets:
            task_masks = targets['task_masks']
            
            # List of dicts 형태: [{'surface': True, 'depth': False}, ...]
            if isinstance(task_masks, list):
                for i, mask in enumerate(task_masks):
                    if isinstance(mask, dict) and mask.get(task_name, False):
                        task_indices.append(i)
            
            # Single dict 형태: {'surface': True, 'depth': False}
            elif isinstance(task_masks, dict):
                if task_masks.get(task_name, False):
                    # 전체 배치에 해당 태스크가 있다고 가정
                    task_indices = list(range(len(targets.get(f'{task_name}_masks', []))))
        
        # 백업: has_{task_name} 플래그 확인
        elif f'has_{task_name}' in targets:
            has_task = targets[f'has_{task_name}']
            if isinstance(has_task, bool) and has_task:
                # 전체 배치에 해당 태스크가 있다고 가정
                task_indices = list(range(len(targets.get(f'{task_name}_masks', []))))
            elif isinstance(has_task, (list, tuple)):
                # Per-sample has 플래그
                task_indices = [i for i, has in enumerate(has_task) if has]
        
        return task_indices

    def _compute_cross_task_consistency_loss(self, 
                                           predictions: Dict, 
                                           targets: Dict,
                                           task_mask: Dict[str, bool]) -> torch.Tensor:
        """Compute MTPSL cross-task consistency loss."""
        embeddings = predictions['cross_task_embeddings']
        device = next(iter(embeddings.values())).device
        
        total_loss = torch.tensor(0.0, device=device)
        loss_count = 0
        
        # Cross-task consistency: surface ↔ depth
        if 'surface_pred_embedding' in embeddings and 'depth_gt_embedding' in embeddings:
            surface_pred = embeddings['surface_pred_embedding']
            depth_gt = embeddings['depth_gt_embedding']
            
            cosine_sim = F.cosine_similarity(surface_pred, depth_gt, dim=1, eps=1e-12)
            loss = 1.0 - cosine_sim.mean()
            total_loss = total_loss + loss
            loss_count += 1
        
        if 'depth_pred_embedding' in embeddings and 'surface_gt_embedding' in embeddings:
            depth_pred = embeddings['depth_pred_embedding']
            surface_gt = embeddings['surface_gt_embedding']
            
            cosine_sim = F.cosine_similarity(depth_pred, surface_gt, dim=1, eps=1e-12)
            loss = 1.0 - cosine_sim.mean()
            total_loss = total_loss + loss
            loss_count += 1
        
        return total_loss / max(loss_count, 1) 