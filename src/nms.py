"""
Non-Maximum Suppression (NMS) for Object Detection

원본 DSPNet의 evaluation 방식을 참고한 정확한 NMS 구현.
IoU 기반 중복 제거와 confidence threshold 적용.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of bounding boxes (원본 DSPNet 방식).
    
    Args:
        box1: (N, 4) boxes [x1, y1, x2, y2]
        box2: (M, 4) boxes [x1, y1, x2, y2]
        
    Returns:
        IoU matrix (N, M)
    """
    # Expand dimensions for broadcasting
    box1 = box1.unsqueeze(1)  # (N, 1, 4)
    box2 = box2.unsqueeze(0)  # (1, M, 4)
    
    # Calculate intersection coordinates
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])
    
    # Calculate intersection area
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Calculate union area
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = area1 + area2 - inter_area
    
    # Calculate IoU
    iou = inter_area / torch.clamp(union_area, min=1e-8)
    
    return iou


def single_class_nms(boxes: torch.Tensor,
                    scores: torch.Tensor,
                    iou_threshold: float = 0.45) -> torch.Tensor:
    """
    Apply NMS for single class (원본 DSPNet 방식).
    
    Args:
        boxes: (N, 4) boxes [x1, y1, x2, y2]
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    
    # Sort by scores in descending order
    _, sorted_indices = scores.sort(descending=True)
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Take the box with highest score
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx.item())
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current_idx:current_idx+1]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = bbox_iou(current_box, remaining_boxes)[0]
        
        # Keep boxes with IoU <= threshold
        keep_mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]
    
    return torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)


def decode_ssd_predictions(cls_logits: torch.Tensor,
                          bbox_deltas: torch.Tensor,
                          anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode SSD predictions to actual boxes and scores (원본 DSPNet 방식).
    
    Args:
        cls_logits: (B, num_anchors, num_classes) classification logits
        bbox_deltas: (B, num_anchors, 5) regression deltas [dx, dy, dw, dh, distance]
        anchors: (num_anchors, 4) anchor boxes [x1, y1, x2, y2]
        
    Returns:
        Tuple of (decoded_boxes, class_scores)
        - decoded_boxes: (B, num_anchors, 4) decoded boxes
        - class_scores: (B, num_anchors, num_classes) class probabilities
    """
    batch_size = cls_logits.size(0)
    num_anchors = cls_logits.size(1)
    num_classes = cls_logits.size(2)
    
    # Convert anchors to center format for decoding
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h
    
    # Decode bounding box deltas
    dx = bbox_deltas[:, :, 0]  # (B, num_anchors)
    dy = bbox_deltas[:, :, 1]
    dw = bbox_deltas[:, :, 2]
    dh = bbox_deltas[:, :, 3]
    
    # Apply transformations (SSD decoding)
    pred_cx = dx * anchor_w + anchor_cx
    pred_cy = dy * anchor_h + anchor_cy
    pred_w = torch.exp(dw) * anchor_w
    pred_h = torch.exp(dh) * anchor_h
    
    # Convert back to corner format
    pred_x1 = pred_cx - 0.5 * pred_w
    pred_y1 = pred_cy - 0.5 * pred_h
    pred_x2 = pred_cx + 0.5 * pred_w
    pred_y2 = pred_cy + 0.5 * pred_h
    
    decoded_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
    
    # Apply softmax to get class probabilities
    class_scores = F.softmax(cls_logits, dim=-1)
    
    return decoded_boxes, class_scores


def multiclass_nms(boxes: torch.Tensor,
                  scores: torch.Tensor,
                  classes: torch.Tensor,
                  conf_threshold: float = 0.5,
                  iou_threshold: float = 0.45,
                  max_detections: int = 100) -> Dict[str, torch.Tensor]:
    """
    Apply multi-class NMS (원본 DSPNet 방식).
    
    Args:
        boxes: (N, 4) detection boxes [x1, y1, x2, y2]
        scores: (N,) confidence scores
        classes: (N,) class predictions
        conf_threshold: Confidence threshold for filtering
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections
        
    Returns:
        Dictionary with filtered results
    """
    # Filter by confidence threshold
    conf_mask = scores > conf_threshold
    if conf_mask.sum() == 0:
        return {
            'boxes': torch.empty(0, 4, device=boxes.device),
            'scores': torch.empty(0, device=scores.device),
            'classes': torch.empty(0, dtype=torch.long, device=classes.device),
            'num_detections': 0
        }
    
    filtered_boxes = boxes[conf_mask]
    filtered_scores = scores[conf_mask]
    filtered_classes = classes[conf_mask]
    
    # Apply NMS per class
    final_keep_indices = []
    unique_classes = filtered_classes.unique()
    
    for cls_id in unique_classes:
        if cls_id == 0:  # Skip background class
            continue
        
        # Get detections for this class
        cls_mask = filtered_classes == cls_id
        cls_boxes = filtered_boxes[cls_mask]
        cls_scores = filtered_scores[cls_mask]
        cls_indices = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
        
        # Apply single-class NMS
        keep_indices = single_class_nms(cls_boxes, cls_scores, iou_threshold)
        
        # Map back to global indices
        global_keep_indices = cls_indices[keep_indices]
        final_keep_indices.append(global_keep_indices)
    
    if final_keep_indices:
        # Concatenate all kept indices
        all_keep_indices = torch.cat(final_keep_indices, dim=0)
        
        # Get final results
        final_boxes = filtered_boxes[all_keep_indices]
        final_scores = filtered_scores[all_keep_indices]
        final_classes = filtered_classes[all_keep_indices]
        
        # Sort by scores and limit to max_detections
        if len(final_scores) > max_detections:
            _, top_indices = final_scores.topk(max_detections)
            final_boxes = final_boxes[top_indices]
            final_scores = final_scores[top_indices]
            final_classes = final_classes[top_indices]
    else:
        # No detections after NMS
        final_boxes = torch.empty(0, 4, device=boxes.device)
        final_scores = torch.empty(0, device=scores.device)
        final_classes = torch.empty(0, dtype=torch.long, device=classes.device)
    
    return {
        'boxes': final_boxes,
        'scores': final_scores,
        'classes': final_classes,
        'num_detections': len(final_boxes)
    }


def batch_nms(cls_logits: torch.Tensor,
              bbox_deltas: torch.Tensor,
              anchors: torch.Tensor,
              conf_threshold: float = 0.5,
              nms_threshold: float = 0.45,
              max_detections: int = 100) -> List[Dict]:
    """
    Apply NMS to a batch of predictions (원본 DSPNet 방식).
    
    Args:
        cls_logits: (B, num_anchors, num_classes) classification logits
        bbox_deltas: (B, num_anchors, 5) regression deltas
        anchors: (num_anchors, 4) anchor boxes
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold
        max_detections: Maximum detections per image
        
    Returns:
        List of detection results for each image in batch
    """
    # Decode predictions
    decoded_boxes, class_scores = decode_ssd_predictions(cls_logits, bbox_deltas, anchors)
    
    batch_results = []
    batch_size = cls_logits.size(0)
    
    for b in range(batch_size):
        boxes = decoded_boxes[b]  # (num_anchors, 4)
        scores = class_scores[b]  # (num_anchors, num_classes)
        
        # Get best class and confidence for each box
        max_scores, max_classes = scores.max(dim=1)
        
        # Apply multiclass NMS
        result = multiclass_nms(
            boxes, max_scores, max_classes,
            conf_threshold, nms_threshold, max_detections
        )
        
        # Add distance predictions if available
        if bbox_deltas.size(2) > 4:
            # Extract distance predictions for kept detections
            # This would require mapping back to original indices
            result['distances'] = torch.zeros(result['num_detections'], device=boxes.device)
        
        batch_results.append(result)
    
    return batch_results


class DSPNetPostProcessor:
    """
    Post-processing pipeline for DSPNet detection outputs (원본 DSPNet 방식).
    """
    
    def __init__(self,
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.45,
                 max_detections: int = 100,
                 class_names: Optional[List[str]] = None):
        """
        Initialize DSPNet post-processor.
        
        Args:
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold  
            max_detections: Maximum detections per image
            class_names: List of class names
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.class_names = class_names
        
        print(f"✅ DSPNet Post-Processor 초기화:")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   NMS threshold: {nms_threshold}")
        print(f"   Max detections: {max_detections}")
    
    def __call__(self,
                 cls_logits: torch.Tensor,
                 bbox_deltas: torch.Tensor,
                 anchors: torch.Tensor) -> List[Dict]:
        """
        Apply post-processing to model outputs.
        
        Args:
            cls_logits: Classification logits
            bbox_deltas: Bounding box regression deltas
            anchors: Anchor boxes
            
        Returns:
            List of detection results
        """
        return batch_nms(
            cls_logits, bbox_deltas, anchors,
            self.conf_threshold, self.nms_threshold, self.max_detections
        )
    
    def format_results(self, results: List[Dict]) -> List[Dict]:
        """
        Format results with class names and additional info.
        
        Args:
            results: Raw detection results
            
        Returns:
            Formatted results with class names
        """
        formatted_results = []
        
        for result in results:
            formatted_result = {
                'boxes': result['boxes'].cpu().numpy(),
                'scores': result['scores'].cpu().numpy(),
                'classes': result['classes'].cpu().numpy(),
                'num_detections': result['num_detections']
            }
            
            # Add distances if available
            if 'distances' in result:
                formatted_result['distances'] = result['distances'].cpu().numpy()
            
            # Add class names if available
            if self.class_names:
                formatted_result['class_names'] = [
                    self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}'
                    for cls in formatted_result['classes']
                ]
            
            formatted_results.append(formatted_result)
        
        return formatted_results


def test_nms():
    """Test NMS functions with dummy data."""
    print("🧪 DSPNet NMS 테스트")
    print("=" * 50)
    
    # Create dummy predictions
    batch_size, num_anchors, num_classes = 2, 1000, 27
    device = torch.device('cpu')
    
    # Random predictions
    cls_logits = torch.randn(batch_size, num_anchors, num_classes)
    bbox_deltas = torch.randn(batch_size, num_anchors, 5)
    anchors = torch.rand(num_anchors, 4) * 512  # Random anchors in 512x512 image
    
    # Ensure anchors are in valid format (x1 < x2, y1 < y2)
    anchors[:, 2] = torch.max(anchors[:, 2], anchors[:, 0] + 1)
    anchors[:, 3] = torch.max(anchors[:, 3], anchors[:, 1] + 1)
    anchors[:, 2] = torch.min(anchors[:, 2], torch.tensor(512.0))
    anchors[:, 3] = torch.min(anchors[:, 3], torch.tensor(512.0))
    
    print(f"📊 테스트 데이터:")
    print(f"   Batch size: {batch_size}")
    print(f"   Anchors: {num_anchors:,}")
    print(f"   Classes: {num_classes}")
    print(f"   Anchor range: x=[{anchors[:, 0].min():.1f}, {anchors[:, 2].max():.1f}], "
          f"y=[{anchors[:, 1].min():.1f}, {anchors[:, 3].max():.1f}]")
    
    # Create post-processor
    post_processor = DSPNetPostProcessor(
        conf_threshold=0.3,
        nms_threshold=0.45,
        max_detections=50
    )
    
    # Apply post-processing
    results = post_processor(cls_logits, bbox_deltas, anchors)
    
    print(f"\n🎯 NMS 결과:")
    print(f"   배치 크기: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"   이미지 {i}: {result['num_detections']}개 검출")
        if result['num_detections'] > 0:
            print(f"      평균 신뢰도: {result['scores'].mean():.3f}")
            print(f"      최고 신뢰도: {result['scores'].max():.3f}")
            print(f"      클래스 분포: {torch.bincount(result['classes']).nonzero().squeeze()}")
    
    # Test IoU calculation
    print(f"\n🔍 IoU 계산 테스트:")
    test_boxes1 = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32)
    test_boxes2 = torch.tensor([[15, 15, 55, 55], [100, 100, 150, 150]], dtype=torch.float32)
    
    ious = bbox_iou(test_boxes1, test_boxes2)
    print(f"   테스트 박스 IoU: {ious}")
    
    print(f"\n✅ NMS 테스트 완료!")
    
    return results


if __name__ == "__main__":
    test_nms() 