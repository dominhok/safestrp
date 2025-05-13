import torch
import math
from typing import List, Tuple

class AnchorGenerator:
    def __init__(self, 
                 image_size: Tuple[int, int], 
                 feature_map_sizes: List[Tuple[int, int]],
                 min_sizes: List[float], # Anchor scale for each feature map
                 max_sizes: List[float] | None = None, # Optional, for SSD style anchors from min_size to sqrt(min_size * max_size)
                 aspect_ratios: List[List[float]] | None = None, # Aspect ratios for each feature map level
                 anchors_per_location_list: List[int] | None = None, # Directly specifies num anchors per loc, if not using aspect_ratios
                 clip: bool = True,
                 **kwargs):
        """
        Args:
            image_size (Tuple[int, int]): Input image size (height, width).
            feature_map_sizes (List[Tuple[int, int]]): List of feature map sizes (height, width) from which to generate anchors.
            min_sizes (List[float]): List of anchor base scales for each feature map.
            max_sizes (List[float] | None): Optional list of max sizes for SSD style anchor generation.
                                           If provided, for each feature map i, an additional anchor with scale
                                           sqrt(min_sizes[i] * max_sizes[i]) and aspect ratio 1 is generated.
            aspect_ratios (List[List[float]] | None): List of aspect ratios (width/height) for each feature map level.
                                                    For example, [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], ...].
                                                    If None, anchors_per_location_list must be used for some default behavior or a simpler anchor set.
            anchors_per_location_list (List[int] | None): If aspect_ratios is None, this list can directly specify the number of
                                                          anchors to generate per location for each feature map. 
                                                          The exact generation logic would need to be defined.
            clip (bool): Whether to clip anchor boxes to stay within image boundaries.
        """
        super().__init__()
        self.image_size = image_size
        self.feature_map_sizes = feature_map_sizes
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios
        self.anchors_per_location_list = anchors_per_location_list # Corresponds to config.model.anchors_per_location_list
        self.clip = clip

        if self.aspect_ratios is None and self.anchors_per_location_list is None:
            # Default to a simple set of aspect ratios if nothing is provided
            # This makes sure len(self.aspect_ratios_per_level[i]) corresponds to self.anchors_per_location_list[i]
            # For example, config's anchors_per_location_list = [4, 6, 6, 6, 4, 4]
            # We would need to define what those 4 or 6 anchors are.
            # A common setup: 1.0, 2.0, 0.5, plus larger 1.0 scale if max_sizes used.
            # Or specific predefined ratios for each count.
            # For now, let's assume if aspect_ratios is None, a default is constructed based on anchors_per_location_list later.
            print("Warning: aspect_ratios is None. Anchor generation might be based on anchors_per_location_list with default ratios.")


        # self.priors will store all generated anchor boxes
        self.priors = self._generate_all_priors()

    def _generate_priors_for_feature_map(self, 
                                         feature_map_size: Tuple[int, int], 
                                         min_size: float,
                                         max_size: float | None,
                                         aspect_ratios_fm: List[float] | None,
                                         num_anchors_per_loc: int | None) -> torch.Tensor:
        """
        Generates anchor boxes for a single feature map.
        """
        fh, fw = feature_map_size
        img_h, img_w = self.image_size
        
        scale_y = float(img_h) / fh
        scale_x = float(img_w) / fw

        boxes = []
        # Generate anchors based on aspect ratios
        if aspect_ratios_fm:
            # First anchor: min_size with aspect_ratio 1.0
            base_h_ar1 = min_size / math.sqrt(1.0)
            base_w_ar1 = min_size * math.sqrt(1.0)
            boxes.append([-base_w_ar1 / 2., -base_h_ar1 / 2., base_w_ar1 / 2., base_h_ar1 / 2.])

            # Anchor with scale sqrt(min_size * max_size) and aspect ratio 1.0
            if max_size:
                h_prime = w_prime = math.sqrt(min_size * max_size)
                boxes.append([-w_prime / 2., -h_prime / 2., w_prime / 2., h_prime / 2.])
            
            # Other aspect ratios
            for ar_val in aspect_ratios_fm:
                if ar_val == 1.0: # Already handled min_size with AR 1.0
                    continue
                h = min_size / math.sqrt(ar_val)
                w = min_size * math.sqrt(ar_val)
                boxes.append([-w / 2., -h / 2., w / 2., h / 2.])
        
        # If no aspect_ratios are given but num_anchors_per_loc is, create some default shapes
        # This part needs careful definition based on how anchors_per_location_list is meant to be used
        elif num_anchors_per_loc:
            # Placeholder: create num_anchors_per_loc anchors with AR 1.0 and slightly varying sizes
            # This logic needs to match the expectations of ssd_depth.py and config.yaml
            base_w = base_h = min_size
            if not boxes: # Add base anchor if not already added by aspect_ratios_fm
                 boxes.append([-base_w / 2., -base_h / 2., base_w / 2., base_h / 2.]) 
            
            if max_size and len(boxes) < num_anchors_per_loc: 
                 h_prime = w_prime = math.sqrt(min_size * max_size)
                 boxes.append([-w_prime / 2., -h_prime / 2., w_prime / 2., h_prime / 2.])
            
            # Fill remaining with some variations if needed
            # This default logic is very basic and likely needs refinement.
            # A more robust approach for N anchors would define specific ARs/scales for N.
            # E.g. if num_anchors_per_loc = 4, use ARs [1.0, 2.0, 0.5] for min_size, and the sqrt(min*max) anchor.
            # Current code with aspect_ratios_fm is more standard for SSD.
            # This fallback is a very rough placeholder.
            idx = 0
            while len(boxes) < num_anchors_per_loc:
                # Add slight variations to base size for remaining anchors
                # This is just a placeholder, a more structured approach (e.g. predefined ARs/scales) is better
                mod_scale = 1.0 + (idx +1) * 0.2 
                mod_w = base_w * mod_scale
                mod_h = base_h * mod_scale # Assuming AR 1.0 for these fillers
                # Avoid duplicate
                is_duplicate = False
                for b in boxes:
                    if abs(b[2] - mod_w/2.0) < 1e-4 and abs(b[3] - mod_h/2.0) < 1e-4 : # simplified check
                        is_duplicate = True
                        break
                if not is_duplicate:
                    boxes.append([-mod_w / 2., -mod_h / 2., mod_w / 2., mod_h / 2.])
                idx +=1
                if idx > 10: # Safety break for placeholder
                    print(f"Warning: Could not generate {num_anchors_per_loc} distinct anchors with placeholder logic. Generated {len(boxes)}.")
                    break


        else:
            raise ValueError("Either aspect_ratios_fm or num_anchors_per_loc must be provided and non-empty.")

        anchor_templates = torch.tensor(boxes) # (num_anchors_at_loc, 4) in (template_xmin,ymin,xmax,ymax) relative to center 0,0

        grid_y = torch.arange(fh, dtype=torch.float32) + 0.5 
        grid_x = torch.arange(fw, dtype=torch.float32) + 0.5
        
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij') # fh, fw

        center_y = mesh_y * scale_y
        center_x = mesh_x * scale_x
        
        num_base_anchors = anchor_templates.size(0)
        
        anchor_centers_x = center_x.unsqueeze(-1).repeat(1, 1, num_base_anchors) 
        anchor_centers_y = center_y.unsqueeze(-1).repeat(1, 1, num_base_anchors) 

        expanded_anchor_templates = anchor_templates.view(1, 1, num_base_anchors, 4)

        boxes_xmin = anchor_centers_x + expanded_anchor_templates[..., 0]
        boxes_ymin = anchor_centers_y + expanded_anchor_templates[..., 1]
        boxes_xmax = anchor_centers_x + expanded_anchor_templates[..., 2]
        boxes_ymax = anchor_centers_y + expanded_anchor_templates[..., 3]

        all_anchors_on_fm = torch.stack([boxes_xmin, boxes_ymin, boxes_xmax, boxes_ymax], dim=-1)
        
        all_anchors_on_fm = all_anchors_on_fm.view(-1, 4)

        if self.clip:
            all_anchors_on_fm[:, 0::2].clamp_(min=0, max=img_w -1) # xmin, xmax
            all_anchors_on_fm[:, 1::2].clamp_(min=0, max=img_h -1) # ymin, ymax
            
        return all_anchors_on_fm

    def _generate_all_priors(self) -> torch.Tensor:
        """
        Generates anchor boxes for all feature maps and concatenates them.
        The order of anchors should match the order of predictions from the network.
        Typically, SSD iterates through feature maps, then spatial locations (row-major), then anchors per location.
        """
        all_priors = []
        for i, fm_size in enumerate(self.feature_map_sizes):
            min_s = self.min_sizes[i]
            max_s = self.max_sizes[i] if self.max_sizes and i < len(self.max_sizes) else None
            ars_fm = self.aspect_ratios[i] if self.aspect_ratios and i < len(self.aspect_ratios) else None
            num_anch_loc = self.anchors_per_location_list[i] if self.anchors_per_location_list and i < len(self.anchors_per_location_list) else None

            if ars_fm is None and num_anch_loc is None:
                 raise ValueError(f"For feature map {i}, either aspect_ratios or anchors_per_location_list must be defined.")
            
            # If num_anch_loc is given, it should be consistent with what aspect_ratios + max_size would produce.
            # The _generate_priors_for_feature_map will use aspect_ratios if available, otherwise fallback to num_anch_loc.
            # No explicit check here, the generation logic handles preference.

            priors_fm = self._generate_priors_for_feature_map(
                feature_map_size=fm_size,
                min_size=min_s,
                max_size=max_s,
                aspect_ratios_fm=ars_fm,
                num_anchors_per_loc=num_anch_loc
            )
            all_priors.append(priors_fm)
        
        return torch.cat(all_priors, dim=0) # Concatenate all anchors: (total_num_anchors, 4)

    def get_priors(self) -> torch.Tensor:
        """
        Returns the generated prior boxes.
        Shape: (total_num_anchors, 4) in (xmin, ymin, xmax, ymax) format.
        """
        return self.priors

    def __call__(self) -> torch.Tensor:
        return self.get_priors()


# --- Utility functions for anchor <-> bbox conversion and IoU ---

def box_cxcywh_to_xyxy(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)."""
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    xmin = cx - 0.5 * w
    ymin = cy - 0.5 * h
    xmax = cx + 0.5 * w
    ymax = cy + 0.5 * h
    return torch.stack((xmin, ymin, xmax, ymax), dim=-1)

def box_xyxy_to_cxcywh(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h)."""
    xmin, ymin, xmax, ymax = boxes_xyxy.unbind(-1)
    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + 0.5 * w
    cy = ymin + 0.5 * h
    return torch.stack((cx, cy, w, h), dim=-1)

def calculate_iou_matrix(boxes1_xyxy: torch.Tensor, boxes2_xyxy: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU (Intersection over Union) matrix between two sets of boxes.
    boxes1_xyxy: (N, 4) tensor of N boxes (xmin, ymin, xmax, ymax).
    boxes2_xyxy: (M, 4) tensor of M boxes (xmin, ymin, xmax, ymax).
    Returns:
        iou_matrix: (N, M) tensor where iou_matrix[i, j] is the IoU between boxes1[i] and boxes2[j].
    """
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])  # (N,)
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])  # (M,)

    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])  
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])

    wh_inter = (rb - lt).clamp(min=0) 
    intersection_area = wh_inter[:, :, 0] * wh_inter[:, :, 1]  

    union_area = area1[:, None] + area2[None, :] - intersection_area 
    
    iou = intersection_area / (union_area + 1e-6) 
    return iou


def encode_boxes_ssd(priors_xyxy: torch.Tensor, gt_boxes_xyxy: torch.Tensor, variances: List[float] = [0.1, 0.1, 0.2, 0.2]) -> torch.Tensor:
    """
    Encode ground truth boxes relative to prior boxes for SSD.
    This is used to generate the regression targets for the localization head.

    Args:
        priors_xyxy: Prior boxes (anchors) in (xmin, ymin, xmax, ymax) format. Shape (num_priors, 4).
        gt_boxes_xyxy: Ground truth boxes in (xmin, ymin, xmax, ymax) format, matched to priors. Shape (num_priors, 4).
                       These are the GT boxes that each prior is responsible for predicting.
        variances: Variances to scale the encoded targets, as used in SSD. 
                   Typically [0.1, 0.1, 0.2, 0.2] for (cx, cy, w, h) respectively.

    Returns:
        encoded_boxes: Encoded localization targets. Shape (num_priors, 4).
                       (target_dcx, target_dcy, target_dw, target_dh)
    """
    priors_cxcywh = box_xyxy_to_cxcywh(priors_xyxy)
    gt_cxcywh = box_xyxy_to_cxcywh(gt_boxes_xyxy)

    target_dcx = (gt_cxcywh[:, 0] - priors_cxcywh[:, 0]) / priors_cxcywh[:, 2] / variances[0]
    target_dcy = (gt_cxcywh[:, 1] - priors_cxcywh[:, 1]) / priors_cxcywh[:, 3] / variances[1]
    target_dw = torch.log(torch.clamp(gt_cxcywh[:, 2] / (priors_cxcywh[:, 2] + 1e-6), min=1e-6)) / variances[2]
    target_dh = torch.log(torch.clamp(gt_cxcywh[:, 3] / (priors_cxcywh[:, 3] + 1e-6), min=1e-6)) / variances[3]

    return torch.stack((target_dcx, target_dcy, target_dw, target_dh), dim=1)


def decode_boxes_ssd(pred_loc_deltas: torch.Tensor, priors_xyxy: torch.Tensor, variances: List[float] = [0.1, 0.1, 0.2, 0.2]) -> torch.Tensor:
    """
    Decode SSD localization predictions (deltas) back to absolute box coordinates.
    This is used during inference or post-processing.

    Args:
        pred_loc_deltas: Predicted localization deltas from the model. Shape (batch_size, num_priors, 4) or (num_priors, 4).
                         (pred_dcx, pred_dcy, pred_dw, pred_dh)
        priors_xyxy: Prior boxes (anchors) in (xmin, ymin, xmax, ymax) format. Shape (num_priors, 4).
                     These should be broadcastable to pred_loc_deltas if batch_size > 1.
        variances: Variances used during encoding.

    Returns:
        decoded_boxes_xyxy: Decoded bounding boxes in (xmin, ymin, xmax, ymax) format.
                            Shape will match pred_loc_deltas shape.
    """
    priors_cxcywh = box_xyxy_to_cxcywh(priors_xyxy) 
    
    if pred_loc_deltas.dim() == 3 and priors_cxcywh.dim() == 2:
        priors_cxcywh = priors_cxcywh.unsqueeze(0) 

    pred_cx = priors_cxcywh[..., 0] + pred_loc_deltas[..., 0] * priors_cxcywh[..., 2] * variances[0]
    pred_cy = priors_cxcywh[..., 1] + pred_loc_deltas[..., 1] * priors_cxcywh[..., 3] * variances[1]
    pred_w = priors_cxcywh[..., 2] * torch.exp(pred_loc_deltas[..., 2] * variances[2])
    pred_h = priors_cxcywh[..., 3] * torch.exp(pred_loc_deltas[..., 3] * variances[3])

    decoded_boxes_cxcywh = torch.stack((pred_cx, pred_cy, pred_w, pred_h), dim=-1)
    
    return box_cxcywh_to_xyxy(decoded_boxes_cxcywh)


class AnchorMatcher:
    def __init__(self, 
                 priors_xyxy: torch.Tensor,
                 iou_threshold_positive: float = 0.5,
                 iou_threshold_negative: float = 0.4,
                 allow_low_quality_matches: bool = True,
                 variances: List[float] = [0.1, 0.1, 0.2, 0.2]):
        """
        Matches ground truth boxes to prior anchor boxes.
        Assigns classification labels and regression targets to anchors.

        Args:
            priors_xyxy (torch.Tensor): All prior boxes (anchors) in (xmin, ymin, xmax, ymax) format.
                                     Shape: (num_total_priors, 4).
            iou_threshold_positive (float): IoU threshold above which an anchor is considered a positive match.
            iou_threshold_negative (float): IoU threshold below which an anchor is considered a negative match.
                                           Anchors with IoU between negative and positive are ignored (label -1).
            allow_low_quality_matches (bool): If True, ensures that each ground truth box is matched to at least
                                             one anchor (the one with highest IoU), even if IoU is below threshold_positive.
                                             This is a common practice in SSD.
            variances (List[float]): Variances for encoding box regression targets.
        """
        self.priors_xyxy = priors_xyxy
        self.iou_threshold_positive = iou_threshold_positive
        self.iou_threshold_negative = iou_threshold_negative
        self.allow_low_quality_matches = allow_low_quality_matches
        self.variances = variances

    def match_single_image(self, gt_boxes_xyxy: torch.Tensor, gt_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Matches anchors to ground truth for a single image.

        Args:
            gt_boxes_xyxy (torch.Tensor): Ground truth bounding boxes for a single image.
                                       Shape: (num_gt_boxes, 4) in (xmin, ymin, xmax, ymax) format.
                                       If no GT boxes, an empty tensor.
            gt_labels (torch.Tensor): Ground truth class labels for each GT box.
                                    Shape: (num_gt_boxes,). Labels should be > 0 for objects.
                                    Background class is typically 0 for model output, but GT labels are obj classes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - matched_gt_labels_for_priors: Labels for each prior. Shape (num_total_priors,).
                    - 0 for background (negative match).
                    - Positive integer for object class (positive match).
                    - -1 for ignored priors.
                - matched_gt_boxes_for_priors_encoded: Encoded GT box targets for each prior.
                    Shape (num_total_priors, 4). For background/ignored, can be zeros or ignored during loss.
        """
        num_priors = self.priors_xyxy.size(0)
        
        matched_gt_labels_for_priors = torch.full((num_priors,), 0, dtype=torch.long, device=self.priors_xyxy.device)
        matched_gt_boxes_for_priors_encoded = torch.zeros((num_priors, 4), dtype=torch.float32, device=self.priors_xyxy.device)

        if gt_boxes_xyxy.numel() == 0: 
            return matched_gt_labels_for_priors, matched_gt_boxes_for_priors_encoded

        iou_matrix = calculate_iou_matrix(self.priors_xyxy, gt_boxes_xyxy)

        best_gt_iou_for_prior, best_gt_idx_for_prior = iou_matrix.max(dim=1)
        
        # Ensure gt_labels are for objects (e.g., > 0). Background (0) should not be assigned here.
        # If your gt_labels for objects start from 0, you'll need to adjust.
        # Assuming object gt_labels are 1-indexed or correspond to class indices > 0.
        
        # 1. Match based on IoU threshold (positive)
        # Anchors whose best GT match has IoU >= positive_threshold
        positive_mask = best_gt_iou_for_prior >= self.iou_threshold_positive
        assigned_gt_labels = gt_labels[best_gt_idx_for_prior] # Get labels of these best GTs
        
        # Only assign positive labels if the GT label itself is an object (not background if it were in gt_labels)
        # This check is more relevant if gt_labels could contain a background class.
        # If gt_labels are purely for objects, this might be redundant, but safe.
        # For SSD, target labels for FocalLoss/CrossEntropy are often 0 for bg, 1..N for objects.
        # So if gt_labels are 1..N, they are directly usable.
        # If gt_labels from dataset are 0..N-1 for objects, add 1 before assigning.
        # Let's assume gt_labels are already in the format 1...N for objects.
        matched_gt_labels_for_priors[positive_mask] = assigned_gt_labels[positive_mask] 
        
        gt_boxes_for_positive_priors = gt_boxes_xyxy[best_gt_idx_for_prior[positive_mask]]
        priors_for_positive_matches = self.priors_xyxy[positive_mask]
        if priors_for_positive_matches.numel() > 0:
             matched_gt_boxes_for_priors_encoded[positive_mask] = encode_boxes_ssd(
                 priors_for_positive_matches, gt_boxes_for_positive_priors, self.variances
             )

        # 2. Handle low IoU matches (negative and ignore)
        # Mark as ignore: negative_thresh <= IoU < positive_thresh
        ignore_mask = (best_gt_iou_for_prior < self.iou_threshold_positive) & \
                      (best_gt_iou_for_prior >= self.iou_threshold_negative)
        matched_gt_labels_for_priors[ignore_mask] = -1 
        # Anchors with IoU < iou_threshold_negative are already background (label 0 by default init)

        # 3. Allow low quality matches (SSD strategy):
        # Ensure each GT box is matched with at least one anchor (the one with highest IoU with it).
        if self.allow_low_quality_matches and gt_boxes_xyxy.size(0) > 0 : # Check if GT boxes exist
            # best_prior_iou_for_gt: (num_gt_boxes,) - IoU value of the best prior for each GT
            # best_prior_idx_for_gt: (num_gt_boxes,) - Index of the best prior for each GT
            # Note: Some GTs might not have any prior with IoU > 0 if they are very small or oddly shaped.
            # Handle cases where no prior overlaps a GT by checking iou_matrix directly for each GT.
            
            for gt_idx in range(gt_boxes_xyxy.size(0)):
                # Find the prior with the highest IoU for this current GT
                max_iou_for_this_gt, best_prior_for_this_gt_idx = iou_matrix[:, gt_idx].max(dim=0)
                
                # If this max_iou is substantial (e.g. > 0, or some small threshold)
                # and this prior is not already assigned to a *better* GT, assign it.
                # SSD's original strategy: For each ground truth, find the prior with the best IoU.
                # Assign this prior to this ground truth. This can override previous assignments.
                # This ensures every GT box has at least one assigned anchor.
                
                # The prior `best_prior_for_this_gt_idx` is now assigned to `gt_labels[gt_idx]`
                # This assignment takes precedence.
                matched_gt_labels_for_priors[best_prior_for_this_gt_idx] = gt_labels[gt_idx]
                
                # Also assign regression target for these forced matches
                matched_gt_boxes_for_priors_encoded[best_prior_for_this_gt_idx] = encode_boxes_ssd(
                    self.priors_xyxy[best_prior_for_this_gt_idx].unsqueeze(0), 
                    gt_boxes_xyxy[gt_idx].unsqueeze(0),            
                    self.variances
                ).squeeze(0)


        return matched_gt_labels_for_priors, matched_gt_boxes_for_priors_encoded

    def __call__(self, gt_boxes_batch: List[torch.Tensor], gt_labels_batch: List[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Matches anchors to ground truth for a batch of images.

        Args:
            gt_boxes_batch (List[torch.Tensor]): List of GT boxes for each image in the batch.
                                              Each tensor is (num_gt_boxes_in_image, 4).
            gt_labels_batch (List[torch.Tensor]): List of GT labels for each image in the batch.
                                               Each tensor is (num_gt_boxes_in_image,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - batch_matched_labels: Labels for priors for each image. Shape (batch_size, num_total_priors).
                - batch_matched_boxes_encoded: Encoded GT boxes for priors for each image. Shape (batch_size, num_total_priors, 4).
        """
        batch_matched_labels = []
        batch_matched_boxes_encoded = []

        for gt_boxes_single_image, gt_labels_single_image in zip(gt_boxes_batch, gt_labels_batch):
            labels, boxes_encoded = self.match_single_image(gt_boxes_single_image, gt_labels_single_image)
            batch_matched_labels.append(labels)
            batch_matched_boxes_encoded.append(boxes_encoded)
        
        return torch.stack(batch_matched_labels, dim=0), torch.stack(batch_matched_boxes_encoded, dim=0)


# Example Usage (for testing, can be removed or moved to a test file)
if __name__ == '__main__':
    # --- AnchorGenerator Example ---
    cfg_image_size = (300, 300) # H, W
    cfg_feature_maps = [
        (38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1) 
    ]
    cfg_min_sizes = [30.0, 60.0, 111.0, 162.0, 213.0, 264.0] 
    cfg_max_sizes = [60.0, 111.0, 162.0, 213.0, 264.0, 315.0] 
    cfg_aspect_ratios = [
        [2.0, 0.5], # For AR 1.0, min_size and sqrt(min*max) are auto-added, so these are additional ARs
        [2.0, 0.5, 3.0, 1/3.0],
        [2.0, 0.5, 3.0, 1/3.0],
        [2.0, 0.5, 3.0, 1/3.0],
        [2.0, 0.5],
        [2.0, 0.5]
    ] 
    # With this setup, num anchors per loc:
    # 1 (min_size@AR1) + 1 (max_size@AR1) + len(cfg_aspect_ratios[i])
    # Layer 0: 1 + 1 + 2 = 4
    # Layer 1: 1 + 1 + 4 = 6
    
    print("Initializing AnchorGenerator with explicit aspect_ratios (excluding 1.0 initially)...")
    anchor_gen = AnchorGenerator(
        image_size=cfg_image_size,
        feature_map_sizes=cfg_feature_maps,
        min_sizes=cfg_min_sizes,
        max_sizes=cfg_max_sizes,
        aspect_ratios=cfg_aspect_ratios, # AR 1.0 handled internally
        clip=True
    )
    all_priors_tensor = anchor_gen.get_priors()
    print(f"Generated {all_priors_tensor.shape[0]} total priors for SSD300-like config.")
    # Expected for SSD300: (38*38*4) + (19*19*6) + (10*10*6) + (5*5*6) + (3*3*4) + (1*1*4) = 8732 anchors
    # (1444*4) + (361*6) + (100*6) + (25*6) + (9*4) + (1*4)
    # = 5776 + 2166 + 600 + 150 + 36 + 4 = 8732. Correct.


    # Example for safestrp config (using its anchors_per_location_list implicitly via aspect_ratios)
    current_cfg_image_size_safestrp = (512, 1024) # H, W (from config.data.image_size: [W,H])
    
    # Feature map sizes for image (512, 1024) -> H, W
    # res3_reduced: (512/8, 1024/8) = (64, 128)
    # res4_reduced: (512/16, 1024/16) = (32, 64)
    # extra1: (32/2, 64/2) = (16, 32) (assuming stride 2 for extra layers)
    # extra2: (16/2, 32/2) = (8, 16)
    # extra3: (8/2, 16/2) = (4, 8)
    # extra4: (4/2, 8/2) = (2, 4)
    cfg_fm_sizes_safestrp = [
        (64, 128), (32, 64), (16, 32), (8, 16), (4, 8), (2, 4)
    ]
    num_feature_maps_safestrp = len(cfg_fm_sizes_safestrp)

    # Define scales s_k = s_min + (s_max - s_min) * (k-1) / (m-1)
    # Let s_min be 10% of image shorter side, s_max be 90%
    s_min_ratio = 0.10 
    s_max_ratio = 0.90 
    shorter_side_safestrp = min(current_cfg_image_size_safestrp)

    cfg_min_anchor_sizes_srp = [ (s_min_ratio + (s_max_ratio - s_min_ratio) * i / (num_feature_maps_safestrp -1)) * shorter_side_safestrp 
                               for i in range(num_feature_maps_safestrp)]
    
    cfg_max_anchor_sizes_srp = []
    for i in range(num_feature_maps_safestrp):
        if i + 1 < num_feature_maps_safestrp:
            # s_k+1 for the sqrt(s_k * s_k+1) anchor
            s_k_plus_1 = (s_min_ratio + (s_max_ratio - s_min_ratio) * (i+1) / (num_feature_maps_safestrp -1)) * shorter_side_safestrp
            cfg_max_anchor_sizes_srp.append(s_k_plus_1) # This is effectively s_{k+1} for the formula sqrt(s_k * s_{k+1})
        else:
            # For the last layer, s_{m+1} is often taken as 1.05 * image_dim or similar
            # Let's use a scale slightly larger than the last s_k, e.g., s_m * 1.05, relative to shorter_side
             cfg_max_anchor_sizes_srp.append(cfg_min_anchor_sizes_srp[-1] * 1.05)


    # Define aspect_ratios to match config.model.anchors_per_location_list = [4, 6, 6, 6, 4, 4]
    # Number of anchors from ARs = total_anchors_per_loc - 2 (for min_size@AR1 and max_size@AR1)
    # For 4 anchors: 4-2 = 2 ARs (e.g. [2.0, 0.5])
    # For 6 anchors: 6-2 = 4 ARs (e.g. [2.0, 0.5, 3.0, 1/3.0])
    safestrp_ars = [
        [2.0, 0.5],                         
        [2.0, 0.5, 3.0, 1.0/3.0],           
        [2.0, 0.5, 3.0, 1.0/3.0],           
        [2.0, 0.5, 3.0, 1.0/3.0],           
        [2.0, 0.5],                         
        [2.0, 0.5]                          
    ]

    print("\\nInitializing AnchorGenerator for safestrp config...")
    anchor_gen_safestrp = AnchorGenerator(
        image_size=current_cfg_image_size_safestrp,
        feature_map_sizes=cfg_fm_sizes_safestrp,
        min_sizes=cfg_min_anchor_sizes_srp,
        max_sizes=cfg_max_anchor_sizes_srp, # This is s_{k+1} for sqrt(s_k * s_{k+1})
        aspect_ratios=safestrp_ars,
        clip=True
    )
    all_priors_tensor_srp = anchor_gen_safestrp.get_priors()
    print(f"Generated {all_priors_tensor_srp.shape[0]} total priors for safestrp config.")
    # Expected total anchors based on config.model.anchors_per_location_list = [4,6,6,6,4,4]
    # (64*128*4) + (32*64*6) + (16*32*6) + (8*16*6) + (4*8*4) + (2*4*4)
    # = 32768 + 12288 + 3072 + 768 + 128 + 32 = 49056
    # The _generate_priors_for_feature_map needs to be accurate for this.
    # Actual from current generator: (num ARs + 2) * fh * fw
    # For first layer: (2+2)*64*128 = 4 * 8192 = 32768. Correct.
    # For second layer: (4+2)*32*64 = 6 * 2048 = 12288. Correct.
    # So the total should be 49056.

    if all_priors_tensor_srp.shape[0] == 49056:
        print("Safestrp anchor count matches expected 49056.")
    else:
        print(f"Safestrp anchor count MISMATCH: Expected 49056, Got {all_priors_tensor_srp.shape[0]}")


    # --- AnchorMatcher Example ---
    print("\\nInitializing AnchorMatcher...")
    matcher = AnchorMatcher(priors_xyxy=all_priors_tensor_srp, iou_threshold_positive=0.5, iou_threshold_negative=0.4)

    example_gt_boxes_img1 = torch.tensor([
        [100, 100, 200, 200], 
        [50, 50, 150, 150],   
        [700, 400, 800, 500] 
    ], dtype=torch.float32).to(all_priors_tensor_srp.device)
    example_gt_labels_img1 = torch.tensor([1, 2, 1], dtype=torch.long).to(all_priors_tensor_srp.device) 

    matched_labels, matched_boxes_encoded = matcher.match_single_image(example_gt_boxes_img1, example_gt_labels_img1)
    print(f"Matched labels shape for img1: {matched_labels.shape}")
    print(f"Matched encoded boxes shape for img1: {matched_boxes_encoded.shape}")
    
    num_pos_anchors = (matched_labels > 0).sum().item()
    num_neg_anchors = (matched_labels == 0).sum().item()
    num_ign_anchors = (matched_labels == -1).sum().item()
    print(f"Img1: Pos anchors: {num_pos_anchors}, Neg anchors: {num_neg_anchors}, Ign anchors: {num_ign_anchors}, Total: {num_pos_anchors+num_neg_anchors+num_ign_anchors}")
    
    example_gt_boxes_batch = [
        example_gt_boxes_img1,
        torch.tensor([[20, 30, 80, 90]], dtype=torch.float32).to(all_priors_tensor_srp.device), 
        torch.empty((0,4), dtype=torch.float32).to(all_priors_tensor_srp.device) 
    ]
    example_gt_labels_batch = [
        example_gt_labels_img1,
        torch.tensor([3], dtype=torch.long).to(all_priors_tensor_srp.device), 
        torch.empty((0,), dtype=torch.long).to(all_priors_tensor_srp.device)
    ]

    batch_matched_labels, batch_matched_boxes_encoded = matcher(example_gt_boxes_batch, example_gt_labels_batch)
    print(f"\\nBatch matched labels shape: {batch_matched_labels.shape}") 
    print(f"Batch matched encoded boxes shape: {batch_matched_boxes_encoded.shape}")

    if num_pos_anchors > 0:
        iou_with_gt0 = calculate_iou_matrix(all_priors_tensor_srp, example_gt_boxes_img1[0].unsqueeze(0)) 
        best_prior_for_gt0_iou, best_prior_for_gt0_idx = iou_with_gt0.max(dim=0)

        if best_prior_for_gt0_iou.item() > 0.1: # Use a lower threshold just for testing any match
            test_prior_idx = best_prior_for_gt0_idx.item()
            test_prior = all_priors_tensor_srp[test_prior_idx]
            test_gt = example_gt_boxes_img1[0]

            print(f"\\nTesting encode/decode:")
            print(f"Test Prior (xyxy) index {test_prior_idx}: {test_prior}")
            print(f"Test GT (xyxy): {test_gt}")

            encoded = encode_boxes_ssd(test_prior.unsqueeze(0), test_gt.unsqueeze(0))
            print(f"Encoded GT (deltas): {encoded}")

            decoded = decode_boxes_ssd(encoded, test_prior.unsqueeze(0))
            print(f"Decoded Box (xyxy): {decoded.squeeze(0)}")
            print(f"IoU between original GT and decoded: {calculate_iou_matrix(test_gt.unsqueeze(0), decoded).item():.4f}")
        else:
            print("\\nSkipping encode/decode test as no prior significantly matched the first GT box for testing.")
