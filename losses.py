import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: 객체 탐지 손실 함수 (Focal Loss, Smooth L1 Loss)
# TODO: 세그멘테이션 손실 함수 (Cross-Entropy Loss with weighting)
# TODO: 깊이 추정 손실 함수 (SILog Loss or L1/L2 Loss)

class SegmentationLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, ignore_index: int = -100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate segmentation loss.

        Args:
            predictions: Raw output from the model (logits).
                         Shape: (batch_size, num_classes, height, width)
            targets: Ground truth segmentation masks.
                     Shape: (batch_size, height, width)
        
        Returns:
            Calculated cross-entropy loss.
        """
        return self.loss_fn(predictions, targets)

# TODO: 객체 탐지 손실 함수 (Focal Loss, Smooth L1 Loss)
# TODO: 깊이 추정 손실 함수 (SILog Loss or L1/L2 Loss)

# --- 객체 탐지 손실 함수 ---
class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    Original Paper: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Classification logits. Shape (batch_size, num_anchors, num_classes)
            targets: Ground truth class labels. Shape (batch_size, num_anchors)
        Returns:
            Calculated focal loss.
        """
        bce_loss = F.binary_cross_entropy_with_logits(predictions, F.one_hot(targets, num_classes=predictions.size(-1)).float(), reduction='none')
        pt = torch.exp(-bce_loss) # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

class ObjectDetectionLoss(nn.Module):
    def __init__(self, num_classes: int, focal_alpha: float = 0.25, focal_gamma: float = 2.0, smooth_l1_beta: float = 1.0, cls_loss_weight: float = 1.0, loc_loss_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.localization_loss = nn.SmoothL1Loss(beta=smooth_l1_beta, reduction='sum') # Sum over coordinates, mean over samples
        self.cls_loss_weight = cls_loss_weight
        self.loc_loss_weight = loc_loss_weight

    def forward(self, 
                pred_cls_logits: torch.Tensor, 
                pred_loc: torch.Tensor, 
                gt_cls_labels: torch.Tensor, 
                gt_loc: torch.Tensor,
                positive_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate combined object detection loss.

        Args:
            pred_cls_logits: Predicted classification logits. Shape (batch_size, num_anchors, num_classes)
            pred_loc: Predicted localization offsets. Shape (batch_size, num_anchors, 4)
            gt_cls_labels: Ground truth class labels. Shape (batch_size, num_anchors)
            gt_loc: Ground truth localization offsets. Shape (batch_size, num_anchors, 4)
            positive_mask: Boolean tensor indicating positive anchors. Shape (batch_size, num_anchors)
                           If None, all anchors with gt_cls_labels > 0 are considered positive for localization.

        Returns:
            total_loss: Combined weighted loss.
            cls_loss: Calculated classification loss.
            loc_loss: Calculated localization loss.
        """
        valid_cls_mask = gt_cls_labels >= 0
        
        batch_size, num_anchors, _ = pred_cls_logits.shape
        
        selected_pred_cls_logits = pred_cls_logits[valid_cls_mask]
        selected_gt_cls_labels = gt_cls_labels[valid_cls_mask]

        if selected_gt_cls_labels.numel() == 0:
            cls_loss = torch.tensor(0.0, device=pred_cls_logits.device, dtype=pred_cls_logits.dtype)
        else:
            cls_loss = self.classification_loss(selected_pred_cls_logits, selected_gt_cls_labels)

        if positive_mask is None:
            positive_mask = (gt_cls_labels > 0)
        
        num_positives = positive_mask.sum().clamp(min=1).float()
        
        selected_pred_loc = pred_loc[positive_mask]
        selected_gt_loc = gt_loc[positive_mask]

        if selected_pred_loc.numel() == 0:
            loc_loss = torch.tensor(0.0, device=pred_loc.device, dtype=pred_loc.dtype)
        else:
            loc_loss = self.localization_loss(selected_pred_loc, selected_gt_loc) / num_positives

        total_loss = (self.cls_loss_weight * cls_loss) + (self.loc_loss_weight * loc_loss)
        return total_loss, cls_loss, loc_loss

# --- 깊이 추정 손실 함수 ---
class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss for depth estimation.
    Often used in monocular depth estimation tasks.
    Measures the error in log space, invariant to global scaling.
    """
    def __init__(self, variance_focus: float = 0.85, eps: float = 1e-6):
        super().__init__()
        self.variance_focus = variance_focus
        self.eps = eps

    def forward(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Calculate SILog loss.

        Args:
            pred_depth: Predicted depth map. Shape (batch_size, 1, height, width) or (batch_size, height, width)
            gt_depth: Ground truth depth map. Shape (batch_size, 1, height, width) or (batch_size, height, width)
            mask: Optional boolean mask to indicate valid pixels for loss calculation. 
                  Shape (batch_size, 1, height, width) or (batch_size, height, width).
                  If None, all pixels are considered valid.

        Returns:
            Calculated SILog loss.
        """
        pred_depth_safe = pred_depth.clamp(min=self.eps)
        gt_depth_safe = gt_depth.clamp(min=self.eps)

        if mask is not None:
            pred_depth_safe = pred_depth_safe[mask]
            gt_depth_safe = gt_depth_safe[mask]
            if pred_depth_safe.numel() == 0:
                return torch.tensor(0.0, device=pred_depth.device, dtype=pred_depth.dtype)

        log_diff = torch.log(pred_depth_safe) - torch.log(gt_depth_safe)
        
        num_pixels = log_diff.numel()
        if num_pixels == 0:
             return torch.tensor(0.0, device=pred_depth.device, dtype=pred_depth.dtype)

        term1 = torch.sum(log_diff ** 2) / num_pixels
        term2 = (torch.sum(log_diff) ** 2) / (num_pixels ** 2)

        loss = term1 - self.variance_focus * term2
        return loss.sqrt() 