"""
Utility functions for loss computation.

Contains anchor matching, IoU computation, and hard negative mining.
"""

import torch
from typing import Tuple


def iou_match_anchors(gt_boxes: torch.Tensor, anchors: torch.Tensor, 
                     positive_threshold: float = 0.5, 
                     negative_threshold: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    IoU 기반 anchor matching (원본 DSPNet 방식).
    
    Args:
        gt_boxes: Ground truth boxes (N, 4) [x1, y1, x2, y2]
        anchors: Anchor boxes (M, 4) [x1, y1, x2, y2]
        positive_threshold: IoU threshold for positive anchors
        negative_threshold: IoU threshold for negative anchors
        
    Returns:
        tuple: (matched_labels, matched_boxes)
            - matched_labels: (M,) anchor labels (-1: ignore, 0: negative, >0: positive class)
            - matched_boxes: (M, 4) matched ground truth boxes for positive anchors
    """
    # 디바이스 일치 확인 및 수정
    device = anchors.device
    gt_boxes = gt_boxes.to(device)
    
    if len(gt_boxes) == 0:
        # No ground truth boxes
        return torch.zeros(len(anchors), dtype=torch.long, device=device), torch.zeros_like(anchors)
    
    # Compute IoU between all anchors and ground truth boxes
    ious = box_iou(anchors, gt_boxes)  # (M, N)
    
    # Find best matching GT for each anchor
    max_ious, matched_gt_indices = ious.max(dim=1)  # (M,)
    
    # Initialize labels (-1: ignore, 0: negative, >0: positive class)
    matched_labels = torch.full((len(anchors),), -1, dtype=torch.long, device=device)
    
    # Set negative samples (low IoU)
    matched_labels[max_ious < negative_threshold] = 0
    
    # Set positive samples (high IoU)
    positive_mask = max_ious >= positive_threshold
    matched_labels[positive_mask] = 1  # 간단히 class 1로 설정 (실제로는 GT class 사용)
    
    # Get matched boxes for positive anchors
    matched_boxes = gt_boxes[matched_gt_indices]
    
    return matched_labels, matched_boxes


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) boxes [x1, y1, x2, y2]
        boxes2: (M, 4) boxes [x1, y1, x2, y2]
        
    Returns:
        IoU matrix (N, M)
    """
    # 디바이스 일치 확인
    device = boxes1.device
    boxes2 = boxes2.to(device)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection coordinates
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - intersection
    iou = intersection / (union + 1e-6)
    
    return iou


def hard_negative_mining(cls_loss: torch.Tensor, labels: torch.Tensor, 
                        neg_pos_ratio: float = 3.0) -> torch.Tensor:
    """
    Hard negative mining for balanced training.
    
    Args:
        cls_loss: Classification loss for each anchor (N,)
        labels: Anchor labels (N,) (-1: ignore, 0: negative, >0: positive)
        neg_pos_ratio: Ratio of negative to positive samples
        
    Returns:
        Selection mask for training (N,)
    """
    # 디바이스 일치 확인
    device = cls_loss.device
    labels = labels.to(device)
    
    # Count positive samples
    positive_mask = labels > 0
    num_positive = positive_mask.sum().item()
    
    if num_positive == 0:
        # No positive samples, select some hard negatives
        negative_mask = labels == 0
        if negative_mask.sum() > 0:
            neg_loss = cls_loss[negative_mask]
            _, sorted_indices = neg_loss.sort(descending=True)
            num_hard_negatives = min(100, len(sorted_indices))  # Select top 100 hard negatives
            
            hard_neg_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
            neg_indices = torch.nonzero(negative_mask, as_tuple=True)[0]
            hard_neg_mask[neg_indices[sorted_indices[:num_hard_negatives]]] = True
            
            return cls_loss[hard_neg_mask].mean() if hard_neg_mask.sum() > 0 else torch.tensor(0.0, device=device)
        else:
            return torch.tensor(0.0, device=device)
    
    # Select hard negatives based on positive count
    negative_mask = labels == 0
    num_negatives = int(neg_pos_ratio * num_positive)
    
    if negative_mask.sum() == 0 or num_negatives == 0:
        return cls_loss[positive_mask].mean() if positive_mask.sum() > 0 else torch.tensor(0.0, device=device)
    
    # Sort negative losses and select hardest ones
    neg_loss = cls_loss[negative_mask]
    _, sorted_indices = neg_loss.sort(descending=True)
    num_selected_negatives = min(num_negatives, len(sorted_indices))
    
    # Create selection mask
    selected_mask = positive_mask.clone()
    neg_indices = torch.nonzero(negative_mask, as_tuple=True)[0]
    selected_mask[neg_indices[sorted_indices[:num_selected_negatives]]] = True
    
    # Return average loss for selected samples
    return cls_loss[selected_mask].mean() if selected_mask.sum() > 0 else torch.tensor(0.0, device=device) 