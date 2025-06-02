"""
Cross-task projection heads for MTPSL consistency.

Contains CrossTaskProjectionHead for cross-task consistency learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .base import BaseHead


class CrossTaskProjectionHead(BaseHead):
    """
    🔧 MTPSL 원본 방식으로 수정된 Cross-task projection heads.
    
    MTPSL 논문의 실제 구현 방식을 따라:
    1. Task-specific input layers로 각 태스크 특성 보존
    2. 모든 태스크를 동일한 embedding dimension으로 매핑
    3. 공통 embedding space에서 직접 cosine similarity 계산
    """
    
    def __init__(self, 
                 seg_channels: int = 7,
                 depth_channels: int = 1,
                 embedding_dim: int = 512,  # MTPSL 원본과 동일
                 input_size: Tuple[int, int] = (512, 512),
                 active_task_pairs: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize cross-task projection heads following MTPSL original design.
        
        Args:
            seg_channels: Number of segmentation classes
            depth_channels: Number of depth channels (usually 1)
            embedding_dim: Common embedding dimension (MTPSL 사용 512)
            input_size: Input image size (H, W)
            active_task_pairs: List of task pairs to enable
        """
        super().__init__()
        
        self.seg_channels = seg_channels
        self.depth_channels = depth_channels
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        
        # Active task pairs (MTPSL 방식)
        if active_task_pairs is None:
            self.active_task_pairs = [('surface', 'depth')]
        else:
            self.active_task_pairs = active_task_pairs
        
        print(f"🔧 MTPSL 원본 방식 적용: {self.active_task_pairs}")
        
        # 🔧 MTPSL 원본: Task-specific input channels
        self.task_input_channels = {
            'surface': seg_channels,  # 7 for surface classes
            'depth': depth_channels,  # 1 for depth
        }
        
        # 🔧 MTPSL 원본: Task-specific input layers (pre-processing)
        self.task_input_layers = nn.ModuleDict()
        for task_name, channels in self.task_input_channels.items():
            self.task_input_layers[f'{task_name}_input'] = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        
        # 🔧 MTPSL 원본: Shared mapping function (SegNet encoder 스타일)
        # 모든 태스크가 동일한 embedding으로 매핑됨
        filter_sizes = [64, 128, 256, 512]
        self.shared_encoder = nn.ModuleList()
        
        in_channels = 64  # task-specific input layer 출력
        for out_channels in filter_sizes:
            self.shared_encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = out_channels
        
        # 🔧 Final embedding layer: 512차원 고정 (MTPSL 원본)
        self.final_embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
        
        print(f"✅ MTPSL 원본 방식 CrossTaskProjectionHead:")
        print(f"   임베딩 차원: {embedding_dim} (모든 태스크 공통)")
        print(f"   태스크별 입력 채널: {self.task_input_channels}")
        print(f"   활성 태스크 쌍: {len(self.active_task_pairs)}개")

    def forward(self, 
                seg_gt: Optional[torch.Tensor] = None,
                seg_pred: Optional[torch.Tensor] = None,
                depth_gt: Optional[torch.Tensor] = None,
                depth_pred: Optional[torch.Tensor] = None,
                task_mask: Optional[Dict[str, bool]] = None) -> Dict[str, torch.Tensor]:
        """
        🔧 MTPSL 원본 방식의 forward pass.
        
        모든 태스크를 동일한 embedding dimension으로 매핑하여
        직접 cosine similarity 계산이 가능하도록 함.
        """
        embeddings = {}
        
        if task_mask is None:
            task_mask = {'surface': True, 'depth': True}
        
        # Task data mapping
        task_data = {
            'surface': {'gt': seg_gt, 'pred': seg_pred},
            'depth': {'gt': depth_gt, 'pred': depth_pred}
        }
        
        # 🔧 MTPSL 원본: 각 태스크별로 동일한 pipeline 적용
        for task_name, data in task_data.items():
            if not task_mask.get(task_name, False):
                continue
                
            gt_data = data['gt']
            pred_data = data['pred']
            
            # GT processing
            if gt_data is not None:
                # 🔧 Pre-process GT (MTPSL 원본 방식)
                gt_processed = self._preprocess_gt(gt_data, task_name)
                gt_embedding = self._extract_embedding(gt_processed, task_name)
                embeddings[f'{task_name}_gt_embedding'] = gt_embedding
            
            # Prediction processing  
            if pred_data is not None:
                # 🔧 Pre-process Prediction (MTPSL 원본 방식)
                pred_processed = self._preprocess_pred(pred_data, task_name)
                pred_embedding = self._extract_embedding(pred_processed, task_name)
                embeddings[f'{task_name}_pred_embedding'] = pred_embedding
        
        return embeddings

    def _preprocess_gt(self, gt: torch.Tensor, task: str) -> torch.Tensor:
        """
        🔧 MTPSL 원본의 GT preprocessing.
        """
        if task == 'surface':
            if gt.dim() == 3:  # (B, H, W) class indices
                # Convert to one-hot: (B, H, W) -> (B, C, H, W)
                gt = F.one_hot(gt.long(), self.seg_channels).permute(0, 3, 1, 2).float()
            # MTPSL 원본: ignore -1 labels 처리는 생략 (SafeStrp에서는 불필요)
            return gt
        elif task == 'depth':
            # MTPSL 원본: normalize by max
            gt_normalized = gt / (gt.max() + 1e-12)
            return gt_normalized
        return gt

    def _preprocess_pred(self, pred: torch.Tensor, task: str) -> torch.Tensor:
        """
        🔧 MTPSL 원본의 Prediction preprocessing.
        """
        if task == 'surface':
            # MTPSL 원본: Gumbel-Softmax for hard assignment
            pred_processed = F.gumbel_softmax(pred, dim=1, tau=1, hard=True)
            # Handle potential NaN (MTPSL 원본 방식)
            while torch.isnan(pred_processed.sum()):
                pred_processed = F.gumbel_softmax(pred, dim=1, tau=1, hard=True)
            return pred_processed
        elif task == 'depth':
            # MTPSL 원본: normalize by max
            pred_normalized = pred / (pred.max() + 1e-12)
            return pred_normalized
        return pred

    def _extract_embedding(self, x: torch.Tensor, task: str) -> torch.Tensor:
        """
        🔧 MTPSL 원본: 모든 태스크를 동일한 embedding으로 변환.
        """
        # Task-specific input layer
        x = self.task_input_layers[f'{task}_input'](x)
        
        # Shared encoder (모든 태스크 공통)
        for encoder_block in self.shared_encoder:
            x = encoder_block(x)
        
        # Final embedding (512차원 고정)
        embedding = self.final_embedding(x)
        
        return embedding

    def compute_cross_task_loss(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        🔧 MTPSL 원본 방식의 cross-task consistency loss.
        
        모든 embedding이 동일한 차원(512)이므로 직접 cosine similarity 계산 가능.
        """
        total_loss = 0.0
        loss_count = 0
        
        for task1, task2 in self.active_task_pairs:
            # Cross-task consistency: task1_pred ↔ task2_gt
            task1_pred_key = f'{task1}_pred_embedding'
            task2_gt_key = f'{task2}_gt_embedding'
            
            if task1_pred_key in embeddings and task2_gt_key in embeddings:
                task1_pred_emb = embeddings[task1_pred_key]  # (B, 512)
                task2_gt_emb = embeddings[task2_gt_key]      # (B, 512)
                
                # 🔧 MTPSL 원본: 1 - cosine_similarity
                cosine_sim = F.cosine_similarity(task1_pred_emb, task2_gt_emb, dim=1, eps=1e-12)
                loss = 1.0 - cosine_sim.mean()
                total_loss += loss
                loss_count += 1
            
            # Reverse direction: task2_pred ↔ task1_gt
            task2_pred_key = f'{task2}_pred_embedding'
            task1_gt_key = f'{task1}_gt_embedding'
            
            if task2_pred_key in embeddings and task1_gt_key in embeddings:
                task2_pred_emb = embeddings[task2_pred_key]  # (B, 512)
                task1_gt_emb = embeddings[task1_gt_key]      # (B, 512)
                
                cosine_sim = F.cosine_similarity(task2_pred_emb, task1_gt_emb, dim=1, eps=1e-12)
                loss = 1.0 - cosine_sim.mean()
                total_loss += loss
                loss_count += 1
        
        if loss_count > 0:
            return total_loss / loss_count
        else:
            return torch.tensor(0.0, device=next(iter(embeddings.values())).device)

    def get_embedding_info(self) -> Dict[str, int]:
        """임베딩 정보 반환."""
        return {
            'embedding_dim': self.embedding_dim,
            'task_input_channels': self.task_input_channels,
            'active_pairs': len(self.active_task_pairs)
        }

    def get_active_pairs(self) -> List[Tuple[str, str]]:
        """현재 활성화된 task pair 목록 반환."""
        return self.active_task_pairs.copy()

    def compute_gt_embeddings(self, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        🔧 GT 데이터로부터 embedding 계산 (MTPSL 원본 방식).
        
        Args:
            targets: 타겟 데이터 (surface_masks, depth_maps 등)
            
        Returns:
            GT embeddings dictionary
        """
        gt_embeddings = {}
        
        # Surface GT embedding
        if 'surface_masks' in targets:
            surface_gt = targets['surface_masks']  # (B, H, W) - class indices
            
            # One-hot encoding으로 변환 (B, H, W) -> (B, C, H, W)
            batch_size, height, width = surface_gt.shape
            surface_gt_onehot = torch.zeros(
                batch_size, self.task_input_channels['surface'], height, width,
                device=surface_gt.device
            )
            surface_gt_onehot.scatter_(1, surface_gt.unsqueeze(1), 1.0)
            
            # Task-specific processing
            surface_processed = self.task_input_layers['surface_input'](surface_gt_onehot)
            
            # Shared encoder
            for encoder_block in self.shared_encoder:
                surface_processed = encoder_block(surface_processed)
            
            # Final embedding
            surface_gt_emb = self.final_embedding(surface_processed)
            gt_embeddings['surface_gt_embedding'] = surface_gt_emb
        
        # Depth GT embedding
        if 'depth_maps' in targets:
            depth_gt = targets['depth_maps']  # (B, 1, H, W)
            
            # Task-specific processing
            depth_processed = self.task_input_layers['depth_input'](depth_gt)
            
            # Shared encoder
            for encoder_block in self.shared_encoder:
                depth_processed = encoder_block(depth_processed)
            
            # Final embedding
            depth_gt_emb = self.final_embedding(depth_processed)
            gt_embeddings['depth_gt_embedding'] = depth_gt_emb
        
        return gt_embeddings


def cosine_similarity_loss(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity loss between two embeddings (MTPSL style).
    
    Args:
        embedding1: (B, C)
        embedding2: (B, C)
        
    Returns:
        Cosine similarity loss (scalar)
    """
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(embedding1, embedding2, dim=1, eps=1e-12)
    
    # Convert to loss (1 - cosine_similarity)
    cosine_loss = 1.0 - cosine_sim.mean()
    
    return cosine_loss 