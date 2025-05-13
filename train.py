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

# Import model components from their respective files
from projects.safestrp.resnet_backbone import ResNetBackbone
from projects.safestrp.ssd_depth import DSPNet_Detector
from projects.safestrp.dspnet_seg import DSPNetSegmentationHead
# Import custom loss functions
from projects.safestrp.losses import SegmentationLoss, ObjectDetectionLoss, SILogLoss # MODIFIED

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

def create_dataloaders(config):
    # TODO: Implement your dataset loading logic here
    # This is a placeholder - you'll need to implement your own dataset classes
    print("Placeholder create_dataloaders called. Implement actual dataset loading.")
    if config['training']['batch_size'] > 0 : 
        dummy_images = torch.randn(config['training']['batch_size'], 3, config['data']['image_size'][1], config['data']['image_size'][0])
        
        h, w = config['data']['image_size'][1], config['data']['image_size'][0]
        # Assuming anchors_per_location_list is available and correctly defined
        num_anchors_placeholder = config['model'].get('anchors_per_location_list', [4,6,6,6,4,4])
        
        s0_h, s0_w = h // 8, w // 8
        s1_h, s1_w = h // 16, w // 16
        s2_h, s2_w = s1_h // 2, s1_w // 2
        s3_h, s3_w = s2_h // 2, s2_w // 2
        s4_h, s4_w = s3_h // 2, s3_w // 2
        s5_h, s5_w = s4_h // 2, s4_w // 2

        total_anchors = (s0_h * s0_w * num_anchors_placeholder[0]) + \
                        (s1_h * s1_w * num_anchors_placeholder[1]) + \
                        (s2_h * s2_w * num_anchors_placeholder[2]) + \
                        (s3_h * s3_w * num_anchors_placeholder[3]) + \
                        (s4_h * s4_w * num_anchors_placeholder[4]) + \
                        (s5_h * s5_w * num_anchors_placeholder[5])

        # For dummy_reg_target, last dimension 5 means (dx, dy, dw, dh, depth_target)
        # For gt_cls_labels, values are class indices. 0 for background.
        # num_detection_classes includes background. If config has num_object_classes, add 1.
        num_det_classes_for_dummy = config['loss'].get('detection_num_classes', config['model'].get('num_detection_classes', 20) + 1)

        dummy_cls_target = torch.randint(0, num_det_classes_for_dummy, (config['training']['batch_size'], total_anchors)) 
        dummy_box_target = torch.rand(config['training']['batch_size'], total_anchors, 4) # dx, dy, dw, dh
        dummy_depth_target = torch.rand(config['training']['batch_size'], total_anchors, 1) * 10 # depth
        
        dummy_seg_target = torch.randint(0, config['model']['num_classes'], (config['training']['batch_size'], config['data']['image_size'][1] // 2, config['data']['image_size'][0] // 2))
        
        dummy_targets = {
            'detection_cls': dummy_cls_target,
            'detection_loc': dummy_box_target,
            'depth': dummy_depth_target, # Associated with anchors for this dummy data
            'segmentation': dummy_seg_target
        }
        num_dummy_batches_train = 5 
        num_dummy_batches_val = 2
        train_loader = [(dummy_images, dummy_targets)] * num_dummy_batches_train
        val_loader = [(dummy_images, dummy_targets)] * num_dummy_batches_val
    else:
        train_loader = []
        val_loader = []

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

        res3_reduced_feat, res4_reduced_feat, res5_reduced_feat = backbone(images)
        
        # pred_cls_logits: (batch, num_anchors, num_classes)
        # pred_box_depth: (batch, num_anchors, 5) where last dim is (dx,dy,dw,dh,depth_pred)
        pred_cls_logits, pred_box_depth = det_depth_model(res3_reduced_feat, res4_reduced_feat)
        
        seg_out = seg_model(res3_reduced_feat, res4_reduced_feat, res5_reduced_feat)

        # Separate box predictions and depth predictions
        pred_loc = pred_box_depth[..., :4]
        pred_depth = pred_box_depth[..., 4:] # Keep as (batch, num_anchors, 1)

        # Detection Loss
        # positive_mask might be needed if your ObjectDetectionLoss expects it explicitly
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

            res3_reduced_feat, res4_reduced_feat, res5_reduced_feat = backbone(images)
            pred_cls_logits, pred_box_depth = det_depth_model(res3_reduced_feat, res4_reduced_feat)
            seg_out = seg_model(res3_reduced_feat, res4_reduced_feat, res5_reduced_feat)

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

    backbone = ResNetBackbone(pretrained=config['model']['pretrained']).to(device)
    
    # Detection model setup
    num_det_classes_config = config['model'].get('num_detection_classes', 20) # Number of actual object classes
    # For ObjectDetectionLoss, num_classes is often (num_object_classes + 1 background)
    # However, FocalLoss inside ObjectDetectionLoss handles one-hot encoding based on pred_cls_logits.size(-1)
    # So, DSPNet_Detector's num_classes should be (num_object_classes + 1)
    # And ObjectDetectionLoss's num_classes should match that.
    # Let's assume config['model']['num_detection_classes'] is the total including background if it's used directly by DSPNet_Detector
    # or config['loss']['detection_num_classes'] is the one for the loss function.
    # To be consistent, DSPNet_Detector should output logits for all classes including background.
    
    # Use num_detection_classes from model config for DSPNet_Detector, which should include background
    # This value is also used by ObjectDetectionLoss internally for FocalLoss if not overridden
    
    # Let's ensure num_det_classes for DSPNet_Detector matches what ObjectDetectionLoss expects.
    # ObjectDetectionLoss will use pred_cls_logits.size(-1) if its own num_classes isn't perfectly aligned
    # with ground truth label encoding. The current ObjectDetectionLoss init takes num_classes.
    
    # Use the value from config['loss'] for ObjectDetectionLoss, which should be number of classes including background
    # Use the value from config['model'] for DSPNet_Detector
    
    # If config['model']['num_detection_classes'] is for actual objects, then DSPNet_Detector needs +1 for background.
    # Let's assume config['model']['num_detection_classes'] ALREADY INCLUDES background.
    # And config['loss']['detection_num_classes'] also includes background.
    
    # The num_classes for ObjectDetectionLoss should match the output of the detector head.
    # DSPNet_Detector is initialized with num_classes from config['model']['num_detection_classes']
    
    # Make sure this is the number of classes the detector head outputs (objects + background)
    detector_output_num_classes = config['model'].get('num_detection_classes', 21) # Assuming this includes background


    anchors_per_loc = config['model'].get('anchors_per_location_list', [4, 6, 6, 6, 4, 4])
    if len(anchors_per_loc) != 6:
        raise ValueError("config.model.anchors_per_location_list must have 6 elements.")

    detection_model = DSPNet_Detector(
        num_classes=detector_output_num_classes, 
        anchors_per_location_list=anchors_per_loc
    ).to(device)
    
    segmentation_model = DSPNetSegmentationHead(
        num_classes=config['model']['num_classes'] # Number of segmentation classes
    ).to(device)
    
    train_loader, val_loader = create_dataloaders(config)
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