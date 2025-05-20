import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# torchvision.models is not directly used here anymore, ResNetBackbone handles it.
import argparse
import yaml
from tqdm import tqdm
import os
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
import math # Added for calculations

# Import model components from their respective files
from projects.safestrp.resnet_backbone import ResNetBackbone
from projects.safestrp.ssd_depth import DSPNet_Detector
from projects.safestrp.dspnet_seg import DSPNetSegmentationHead
# Import custom loss functions
from projects.safestrp.losses import SegmentationLoss, ObjectDetectionLoss, SILogLoss # MODIFIED
from projects.safestrp.utils.anchors import AnchorGenerator, AnchorMatcher # Added

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-task Learning Training')
    parser.add_argument('--config', type=str, default='projects/safestrp/config.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_anchor_config(config, image_h, image_w):
    """Prepares configuration for AnchorGenerator based on main config."""
    num_feature_levels = 7 # Consistent with DSPNet_Detector

    # 1. Feature map sizes (derive from image size and backbone structure)
    # (H, W) format for feature maps
    feature_map_sizes = []
    current_h, current_w = image_h, image_w
    strides = [8, 2, 2, 2, 2, 2, 2] # Effective strides for C3,C4,C5, EL1,EL2,EL3,EL4
    # C3: /8. C4: /16. C5: /32. EL1: /64 ... EL4: /256 is wrong.
    # Strides for feature maps for detector:
    # C3 from ResNet Layer2: H/8, W/8
    # C4 from ResNet Layer3: H/16, W/16
    # C5 from ResNet Layer4: H/32, W/32
    # ExtraLayer1 (input C5, stride 2): H/64, W/64
    # ExtraLayer2 (input EL1, stride 2): H/128, W/128
    # ExtraLayer3 (input EL2, stride 2): H/256, W/256
    # ExtraLayer4 (input EL3, stride 2): H/512, W/512
    base_strides = [8, 16, 32, 64, 128, 256, 512]
    for stride_val in base_strides:
        feature_map_sizes.append((math.ceil(image_h / stride_val), math.ceil(image_w / stride_val)))
    
    # 2. Min/Max sizes for anchors (example: SSD-style scale progression)
    # These should ideally come from config or be carefully designed.
    # Using placeholder values or simplified SSD-like calculation for now.
    # dspnet has explicit min_sizes like [30,60,120,180,240,300] for 6 levels
    # For 7 levels, we need 7 min_sizes. Max_sizes are optional for AnchorGenerator
    # (used for sqrt(min*max) anchor).
    default_min_sizes = config['model'].get('anchor_min_sizes', [20.0, 40.0, 80.0, 120.0, 160.0, 220.0, 280.0])
    default_max_sizes = config['model'].get('anchor_max_sizes', [40.0, 80.0, 120.0, 160.0, 220.0, 280.0, 320.0]) 
    # Ensure lists have 7 elements if provided, otherwise use defaults that have 7.
    if len(default_min_sizes) != num_feature_levels:
        print(f"Warning: anchor_min_sizes in config should have {num_feature_levels} elements. Using default.")
        default_min_sizes = [20.0, 40.0, 80.0, 120.0, 160.0, 220.0, 280.0]
    if default_max_sizes and len(default_max_sizes) != num_feature_levels:
        print(f"Warning: anchor_max_sizes in config should have {num_feature_levels} elements if provided. Using default or disabling.")
        default_max_sizes = [40.0, 80.0, 120.0, 160.0, 220.0, 280.0, 320.0]

    # 3. Aspect Ratios (derived from anchors_per_location_list)
    # anchors_per_location_list: e.g. [4,4,6,6,6,4,4] for 7 feature maps
    # Number of anchors = 1 (min@AR1) + (1 if max_size else 0) (max@AR1) + len(aspect_ratios_list_for_level)
    anchor_counts = config['model'].get('anchors_per_location_list', [4, 4, 6, 6, 6, 4, 4])
    if len(anchor_counts) != num_feature_levels:
        print(f"Warning: anchors_per_location_list must have {num_feature_levels} elements. Using default.")
        anchor_counts = [4, 4, 6, 6, 6, 4, 4]

    # Define what ARs to add for a given target count of anchors per location
    # Assumes max_sizes are provided, so 2 base anchors (min@1, sqrt(min*max)@1) are always generated.
    # The list here specifies additional ARs to apply to min_size.
    aspect_ratios_map = {
        2: [],  # Results in 2 anchors if max_sizes are used
        3: [[2.0]], # Results in 2 + 1 = 3 anchors (e.g. AR 2.0 for min_size)
        4: [[2.0, 0.5]], # Results in 2 + 2 = 4 anchors
        5: [[2.0, 0.5, 3.0]],
        6: [[2.0, 0.5, 3.0, 1.0/3.0]]
    }
    aspect_ratios_for_generator = []
    for count in anchor_counts:
        num_additional_ars_needed = count - (2 if default_max_sizes else 1)
        found_map = False
        for key_count, ar_list in aspect_ratios_map.items():
             if len(ar_list) == num_additional_ars_needed and (key_count == count if default_max_sizes else key_count == count -1 ):
                  aspect_ratios_for_generator.append(ar_list)
                  found_map = True
                  break
        if not found_map:
            print(f"Warning: Could not map anchor count {count} to a predefined aspect ratio set. Using empty AR list for this level, resulting in {2 if default_max_sizes else 1} anchors.")
            aspect_ratios_for_generator.append([])

    return {
        'image_size': (image_h, image_w),
        'feature_map_sizes': feature_map_sizes,
        'min_sizes': default_min_sizes,
        'max_sizes': default_max_sizes, # Can be None if not configured
        'aspect_ratios': aspect_ratios_for_generator,
        'clip': config['model'].get('anchors_clip', True),
        'anchors_per_location_list': anchor_counts # For reference
    }

def create_dataloaders(config, anchor_config_for_priors, anchor_matcher):
    print("Placeholder create_dataloaders called. Implement actual dataset loading.")
    batch_size = config['training']['batch_size']
    if batch_size <= 0:
        return [], []

    image_h, image_w = anchor_config_for_priors['image_size']
    dummy_images = torch.randn(batch_size, 3, image_h, image_w)
    
    # Use AnchorGenerator to calculate total_anchors for dummy target shapes
    # This is a bit circular for dummy data, but demonstrates use of AnchorGenerator properties
    # For real data, priors are generated once and used by matcher in dataset __getitem__
    temp_anchor_gen = AnchorGenerator(**anchor_config_for_priors) # Use full config
    total_anchors = temp_anchor_gen.get_priors().shape[0]
    print(f"Total anchors calculated by AnchorGenerator for dummy data shape: {total_anchors}")

    # Check against manual calculation (from previous version for verification)
    # This manual calculation should match AnchorGenerator if config is consistent.
    num_anchors_per_loc_config = anchor_config_for_priors['anchors_per_location_list']
    manual_total_anchors = 0
    for i, fm_size in enumerate(anchor_config_for_priors['feature_map_sizes']):
        manual_total_anchors += fm_size[0] * fm_size[1] * num_anchors_per_loc_config[i]
    
    if total_anchors != manual_total_anchors:
        print(f"Warning: AnchorGenerator total anchors ({total_anchors}) mismatch manual calc ({manual_total_anchors}). Check anchor config consistency.")
        # Fallback to manual for dummy data if mismatch, though ideally this shouldn't happen.
        total_anchors = manual_total_anchors 

    num_det_classes_for_dummy = config['model'].get('num_detection_classes', 21) # Includes background

    dummy_cls_target = torch.randint(0, num_det_classes_for_dummy, (batch_size, total_anchors))
    dummy_box_target = torch.rand(batch_size, total_anchors, 4)
    dummy_depth_target = torch.rand(batch_size, total_anchors, 1) * 10
    
    seg_h_out, seg_w_out = image_h // 2, image_w // 2 # Example output size for segmentation
    dummy_seg_target = torch.randint(0, config['model']['num_classes'], (batch_size, seg_h_out, seg_w_out))
    
    dummy_targets = {
        'detection_cls': dummy_cls_target,
        'detection_loc': dummy_box_target,
        'depth': dummy_depth_target,
        'segmentation': dummy_seg_target
    }
    num_dummy_batches_train = 5
    num_dummy_batches_val = 2
    train_loader = [(dummy_images, dummy_targets)] * num_dummy_batches_train
    val_loader = [(dummy_images, dummy_targets)] * num_dummy_batches_val

    return train_loader, val_loader

# MODIFIED train function signature and body
def train(backbone, det_depth_model, seg_model, train_loader, optimizer, criterion, device, epoch, writer, config):
    backbone.train()
    det_depth_model.train()
    seg_model.train()
    
    epoch_total_loss = 0
    epoch_det_cls_loss = 0
    epoch_det_loc_loss = 0
    epoch_depth_loss = 0
    epoch_seg_loss = 0
    
    loss_weights = config['loss']['weights']
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        gt_det_cls = targets['detection_cls'].to(device).long() # Ensure long type for CrossEntropy/FocalLoss
        gt_det_loc = targets['detection_loc'].to(device)
        gt_depth = targets['depth'].to(device) # (batch, num_anchors, 1)
        gt_seg = targets['segmentation'].to(device).long() # Ensure long type

        c3_feat, c4_feat, c5_feat = backbone(images)
        
        # pred_cls_logits: (batch, num_anchors, num_classes)
        # pred_box_depth: (batch, num_anchors, 5) where last dim is (dx,dy,dw,dh,depth_pred)
        pred_cls_logits, pred_box_depth = det_depth_model(c3_feat, c4_feat, c5_feat)
        
        seg_out = seg_model(c3_feat, c4_feat, c5_feat)

        # Separate box predictions and depth predictions
        pred_loc = pred_box_depth[..., :4]
        pred_depth = pred_box_depth[..., 4:] # Keep as (batch, num_anchors, 1)

        # Detection Loss
        # positive_mask might be needed if your ObjectDetectionLoss expects it explicitly
        # For now, ObjectDetectionLoss handles positive_mask internally based on gt_det_cls > 0
        # For now, ObjectDetectionLoss handles positive_mask internally based on gt_det_cls > 0
        det_total_loss, det_cls_loss, det_loc_loss = criterion['detection'](
            pred_cls_logits, pred_loc, gt_det_cls, gt_det_loc
        )

        # Depth Loss
        # SILogLoss expects (B, 1, H, W) or (B,H,W). Current pred_depth is (B, num_anchors, 1)
        # This requires a re-think of how depth is predicted and targeted if using SILog directly
        # For this example, assuming a placeholder "anchor-wise" depth loss or that targets/preds are reshaped/processed.
        # Let's assume gt_depth and pred_depth are suitable for a simple L1 on anchor depths for now,
        # or SILogLoss needs to be adapted / or depth is predicted per-pixel not per-anchor.
        # For now, let's use a placeholder for depth loss calculation that matches shapes.
        # This is a CRITICAL point: The current SILogLoss is designed for dense depth maps.
        # If depth is predicted per anchor, SILogLoss might not be directly applicable without modification
        # or a different loss like SmoothL1Loss applied to depth component.
        # Using SmoothL1 for depth on anchors as a placeholder:
        if 'depth' in criterion:
             # gt_depth is (B, num_anchors, 1), pred_depth is (B, num_anchors, 1)
             # We need to apply it only to positive anchors for depth, similar to loc loss
             positive_mask_for_depth = (gt_det_cls > 0).unsqueeze(-1).expand_as(pred_depth)
             if positive_mask_for_depth.sum() > 0:
                 depth_loss_val = criterion['depth'](pred_depth[positive_mask_for_depth], gt_depth[positive_mask_for_depth])
             else:
                 depth_loss_val = torch.tensor(0.0, device=device)
        else: # If SILogLoss was intended, this part needs a proper implementation strategy
             depth_loss_val = torch.tensor(0.0, device=device) # Placeholder if no depth criterion

        # Segmentation Loss
        seg_loss_val = criterion['segmentation'](seg_out, gt_seg)

        # Total Weighted Loss
        loss = (loss_weights.get('detection', 1.0) * det_total_loss +
                loss_weights.get('depth', 1.0) * depth_loss_val + # ADDED depth loss
                loss_weights.get('segmentation', 1.0) * seg_loss_val)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_total_loss += loss.item()
        epoch_det_cls_loss += det_cls_loss.item()
        epoch_det_loc_loss += det_loc_loss.item()
        epoch_depth_loss += depth_loss_val.item() # ADDED
        epoch_seg_loss += seg_loss_val.item()
        
        progress_bar.set_postfix(
            loss=loss.item(), 
            det_cls=det_cls_loss.item(), 
            det_loc=det_loc_loss.item(),
            depth=depth_loss_val.item(), # ADDED
            seg=seg_loss_val.item()
        )

        if writer and batch_idx % config['logging']['log_interval'] == 0:
            current_iter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/total_loss', loss.item(), current_iter)
            writer.add_scalar('train/detection_cls_loss', det_cls_loss.item(), current_iter)
            writer.add_scalar('train/detection_loc_loss', det_loc_loss.item(), current_iter)
            writer.add_scalar('train/depth_loss', depth_loss_val.item(), current_iter) # ADDED
            writer.add_scalar('train/segmentation_loss', seg_loss_val.item(), current_iter)
            
    return (epoch_total_loss / len(train_loader) if len(train_loader) > 0 else 0,
            epoch_det_cls_loss / len(train_loader) if len(train_loader) > 0 else 0,
            epoch_det_loc_loss / len(train_loader) if len(train_loader) > 0 else 0,
            epoch_depth_loss / len(train_loader) if len(train_loader) > 0 else 0, # ADDED
            epoch_seg_loss / len(train_loader) if len(train_loader) > 0 else 0)


# MODIFIED validate function signature and body
def validate(backbone, det_depth_model, seg_model, val_loader, criterion, device, epoch, writer, config):
    backbone.eval()
    det_depth_model.eval()
    seg_model.eval()

    epoch_total_loss = 0
    epoch_det_cls_loss = 0
    epoch_det_loc_loss = 0
    epoch_depth_loss = 0
    epoch_seg_loss = 0

    loss_weights = config['loss']['weights']

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            gt_det_cls = targets['detection_cls'].to(device).long()
            gt_det_loc = targets['detection_loc'].to(device)
            gt_depth = targets['depth'].to(device)
            gt_seg = targets['segmentation'].to(device).long()

            c3_feat, c4_feat, c5_feat = backbone(images)
            pred_cls_logits, pred_box_depth = det_depth_model(c3_feat, c4_feat, c5_feat)
            seg_out = seg_model(c3_feat, c4_feat, c5_feat)

            pred_loc = pred_box_depth[..., :4]
            pred_depth = pred_box_depth[..., 4:]

            det_total_loss, det_cls_loss, det_loc_loss = criterion['detection'](
                pred_cls_logits, pred_loc, gt_det_cls, gt_det_loc
            )
            
            if 'depth' in criterion:
                 positive_mask_for_depth = (gt_det_cls > 0).unsqueeze(-1).expand_as(pred_depth)
                 if positive_mask_for_depth.sum() > 0:
                     depth_loss_val = criterion['depth'](pred_depth[positive_mask_for_depth], gt_depth[positive_mask_for_depth])
                 else:
                     depth_loss_val = torch.tensor(0.0, device=device)
            else:
                 depth_loss_val = torch.tensor(0.0, device=device)

            seg_loss_val = criterion['segmentation'](seg_out, gt_seg)

            loss = (loss_weights.get('detection', 1.0) * det_total_loss +
                    loss_weights.get('depth', 1.0) * depth_loss_val +
                    loss_weights.get('segmentation', 1.0) * seg_loss_val)

            epoch_total_loss += loss.item()
            epoch_det_cls_loss += det_cls_loss.item()
            epoch_det_loc_loss += det_loc_loss.item()
            epoch_depth_loss += depth_loss_val.item()
            epoch_seg_loss += seg_loss_val.item()
            
            progress_bar.set_postfix(loss=loss.item())

    avg_total_loss = epoch_total_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_det_cls_loss = epoch_det_cls_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_det_loc_loss = epoch_det_loc_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_depth_loss = epoch_depth_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_seg_loss = epoch_seg_loss / len(val_loader) if len(val_loader) > 0 else 0
    
    if writer:
        writer.add_scalar('val/total_loss', avg_total_loss, epoch)
        writer.add_scalar('val/detection_cls_loss', avg_det_cls_loss, epoch)
        writer.add_scalar('val/detection_loc_loss', avg_det_loc_loss, epoch)
        writer.add_scalar('val/depth_loss', avg_depth_loss, epoch)
        writer.add_scalar('val/segmentation_loss', avg_seg_loss, epoch)
        
    return avg_total_loss, avg_det_cls_loss, avg_det_loc_loss, avg_depth_loss, avg_seg_loss


def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Image dimensions from config (H, W for consistency with model internals)
    image_h = config['data']['image_size'][1]
    image_w = config['data']['image_size'][0]

    # --- Anchor Setup ---
    anchor_params = get_anchor_config(config, image_h, image_w)
    anchor_generator = AnchorGenerator(**anchor_params) # Pass all derived params
    priors_xyxy = anchor_generator.get_priors().to(device)
    
    anchor_matcher = AnchorMatcher(
        priors_xyxy=priors_xyxy,
        iou_threshold_positive=config['loss'].get('anchor_iou_positive_threshold', 0.5),
        iou_threshold_negative=config['loss'].get('anchor_iou_negative_threshold', 0.4),
        allow_low_quality_matches=config['loss'].get('anchor_allow_low_quality', True),
        variances=config['model'].get('anchor_variances', [0.1, 0.1, 0.2, 0.2])
    )
    print(f"Generated {priors_xyxy.shape[0]} priors/anchors.")
    # --- End Anchor Setup ---

    backbone = ResNetBackbone(pretrained=config['model']['pretrained']).to(device)
    
    detector_output_num_classes = config['model'].get('num_detection_classes', 21) # Includes background
    # anchors_per_location_list is now sourced from anchor_params for consistency
    anchors_per_loc_list_for_detector = anchor_params['anchors_per_location_list'] 

    detection_model = DSPNet_Detector(
        num_classes=detector_output_num_classes, 
        anchors_per_location_list=anchors_per_loc_list_for_detector
    ).to(device)
    
    segmentation_model = DSPNetSegmentationHead(
        num_classes=config['model']['num_classes'] # Number of segmentation classes
    ).to(device)
    
    # Pass anchor_params and anchor_matcher to dataloaders for real data handling
    # For dummy data, create_dataloaders uses anchor_params to shape targets.
    train_loader, val_loader = create_dataloaders(config, anchor_params, anchor_matcher)
    if not train_loader or not val_loader:
        print("Dataloaders are not initialized. Exiting. Please implement create_dataloaders.")
        return

    # Initialize Loss Functions using config
    loss_config = config.get('loss', {})
    
    # Segmentation Loss
    seg_class_weights_list = loss_config.get('segmentation_class_weights', None)
    seg_class_weights_tensor = torch.tensor(seg_class_weights_list, device=device) if seg_class_weights_list else None
    
    criterion_segmentation = SegmentationLoss(
        weight=seg_class_weights_tensor,
        ignore_index=loss_config.get('segmentation_ignore_index', -100)
    )

    # Object Detection Loss
    # Ensure num_classes for ObjectDetectionLoss matches detector_output_num_classes
    criterion_detection = ObjectDetectionLoss(
        num_classes=detector_output_num_classes, 
        focal_alpha=loss_config.get('focal_loss', {}).get('alpha', 0.25),
        focal_gamma=loss_config.get('focal_loss', {}).get('gamma', 2.0),
        smooth_l1_beta=loss_config.get('smooth_l1_beta', 1.0)
        # cls_loss_weight and loc_loss_weight inside ObjectDetectionLoss are 1.0 by default,
        # the overall detection task weight is applied later.
    )

    # Depth Loss (Using SmoothL1Loss as a placeholder for anchor-wise depth)
    # SILogLoss from losses.py is for dense depth maps. If depth is predicted per anchor and SILog is desired,
    # either the loss needs adaptation, or the network output/target format needs to change.
    # For now, using SmoothL1Loss for the depth component of pred_box_depth
    criterion_depth = nn.SmoothL1Loss(reduction='mean') # Or use SILog if adapted for anchors

    criterion = {
        'detection': criterion_detection,
        'segmentation': criterion_segmentation,
        'depth': criterion_depth # Placeholder for depth per anchor
    }
    
    params = list(backbone.parameters()) + list(detection_model.parameters()) + list(segmentation_model.parameters())
    
    opt_config = config['training']
    if opt_config.get('optimizer', 'sgd').lower() == 'adam':
        optimizer = optim.Adam(params, lr=opt_config['learning_rate'], weight_decay=opt_config.get('weight_decay', 0))
    else: 
        optimizer = optim.SGD(params, lr=opt_config['learning_rate'], momentum=opt_config.get('momentum', 0.9), weight_decay=opt_config.get('weight_decay', 0.0001))
    
    scheduler = MultiStepLR(optimizer, milestones=opt_config.get('lr_milestones', [80, 160, 240]), gamma=opt_config.get('lr_gamma', 0.1))
    
    writer = None
    if config['logging'].get('tensorboard', False):
        log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        writer = SummaryWriter(log_dir)
        print(f"Tensorboard logs will be saved to: {log_dir}")
    
    start_epoch = 0
    output_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')

    if args.resume and args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Resuming from checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            backbone.load_state_dict(checkpoint['backbone_state_dict'])
            detection_model.load_state_dict(checkpoint['detection_model_state_dict'])
            segmentation_model.load_state_dict(checkpoint['segmentation_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict())) 
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"Resumed. Starting from epoch {start_epoch}. Best val loss: {best_val_loss:.4f}")
        else:
            print(f"Checkpoint not found at {args.checkpoint}. Starting from scratch.")

    # Overall loss weights are now fetched from config['loss']['weights']['segmentation']
    # The old wseg is effectively replaced by config['loss']['weights']['segmentation']

    for epoch in range(start_epoch, config['training']['epochs']):
        train_results = train(backbone, detection_model, segmentation_model, train_loader, optimizer, criterion, device, epoch, writer, config)
        train_total_loss, train_det_cls, train_det_loc, train_depth, train_seg = train_results
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"Train Total Loss: {train_total_loss:.4f}, DetCls: {train_det_cls:.4f}, DetLoc: {train_det_loc:.4f}, Depth: {train_depth:.4f}, Seg: {train_seg:.4f}")
        
        val_results = validate(backbone, detection_model, segmentation_model, val_loader, criterion, device, epoch, writer, config)
        val_total_loss, val_det_cls, val_det_loc, val_depth, val_seg = val_results
        print(f"Validation Total Loss: {val_total_loss:.4f}, DetCls: {val_det_cls:.4f}, DetLoc: {val_det_loc:.4f}, Depth: {val_depth:.4f}, Seg: {val_seg:.4f}")
        
        scheduler.step()
        
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            save_path = os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'detection_model_state_dict': detection_model.state_dict(),
                'segmentation_model_state_dict': segmentation_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, save_path)
            print(f"Model saved to {save_path}")

    if writer:
        writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main() 