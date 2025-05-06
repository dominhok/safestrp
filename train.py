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

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-task Learning Training')
    parser.add_argument('--config', type=str, default='projects/safestrp/config.yaml', help='Path to config file') # Default path updated
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
        
        # Determine number of anchors based on image size and detector's expected feature map sizes
        # This is a rough estimation and should be calculated accurately based on model architecture if needed for dummy targets
        # For res3_reduced (H/8, W/8) e.g., (512/8, 1024/8) = (64, 128)
        # For res4_reduced (H/16, W/16) e.g., (512/16, 1024/16) = (32, 64)
        # ... and so on for extra layers.
        # The value 2476 was for a 320x320 input in an older example. Let's use a more generic placeholder or calculate it.
        # For simplicity, using a fixed large enough number for dummy target.
        # A more robust way would be to pass a dummy input through the detector once to get the actual number of anchors.
        num_anchors_placeholder = config['model'].get('anchors_per_location_list', [4,6,6,6,4,4])
        
        # Calculate total anchors based on example image size H=512, W=1024 from config
        # These are example calculations and might need adjustment based on actual model output shapes
        h, w = config['data']['image_size'][1], config['data']['image_size'][0]
        s0_h, s0_w = h // 8, w // 8   # res3_reduced
        s1_h, s1_w = h // 16, w // 16 # res4_reduced
        s2_h, s2_w = s1_h // 2, s1_w // 2 # extra_layer1 from res4_reduced
        s3_h, s3_w = s2_h // 2, s2_w // 2 # extra_layer2
        s4_h, s4_w = s3_h // 2, s3_w // 2 # extra_layer3
        s5_h, s5_w = s4_h // 2, s4_w // 2 # extra_layer4

        total_anchors = (s0_h * s0_w * num_anchors_placeholder[0]) + \
                        (s1_h * s1_w * num_anchors_placeholder[1]) + \
                        (s2_h * s2_w * num_anchors_placeholder[2]) + \
                        (s3_h * s3_w * num_anchors_placeholder[3]) + \
                        (s4_h * s4_w * num_anchors_placeholder[4]) + \
                        (s5_h * s5_w * num_anchors_placeholder[5])

        dummy_cls_target = torch.randint(0, config['model'].get('num_detection_classes', 20) + 1, (config['training']['batch_size'], total_anchors)) 
        dummy_reg_target = torch.rand(config['training']['batch_size'], total_anchors, 5)
        dummy_seg_target = torch.randint(0, config['model']['num_classes'], (config['training']['batch_size'], config['data']['image_size'][1] // 2, config['data']['image_size'][0] // 2))
        
        dummy_targets = {
            'cls': dummy_cls_target,
            'box_and_depth': dummy_reg_target,
            'segmentation': dummy_seg_target
        }
        # Ensure num_samples is divisible by batch_size for simplicity in dummy loader
        num_dummy_batches_train = 5 
        num_dummy_batches_val = 2
        train_loader = [(dummy_images, dummy_targets)] * num_dummy_batches_train
        val_loader = [(dummy_images, dummy_targets)] * num_dummy_batches_val
    else:
        train_loader = []
        val_loader = []

    return train_loader, val_loader

def train(backbone, det_depth_model, seg_model, train_loader, optimizer, criterion, device, epoch, writer, config, wseg):
    backbone.train()
    det_depth_model.train()
    seg_model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets_cls = targets['cls'].to(device)
        targets_box_depth = targets['box_and_depth'].to(device)
        targets_seg = targets['segmentation'].to(device)

        # resnet_backbone.py returns: res3_reduced, res4_reduced, res5_reduced
        res3_reduced_feat, res4_reduced_feat, res5_reduced_feat = backbone(images)
        
        # DSPNet_Detector now expects res3_reduced (128ch) and res4_reduced (256ch)
        det_cls_pred, det_box_depth_pred = det_depth_model(res3_reduced_feat, res4_reduced_feat)
        
        # DSPNetSegmentationHead expects res3_reduced, res4_reduced, res5_reduced
        seg_out = seg_model(res3_reduced_feat, res4_reduced_feat, res5_reduced_feat)
        
        cls_loss = criterion['cls'](det_cls_pred.view(-1, det_cls_pred.size(-1)), targets_cls.view(-1)) 
        reg_loss = criterion['reg'](det_box_depth_pred, targets_box_depth)
        seg_loss = criterion['segmentation'](seg_out, targets_seg)

        loss = cls_loss + reg_loss + wseg * seg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        progress_bar.set_postfix(loss=loss.item(), cls=cls_loss.item(), reg=reg_loss.item(), seg=seg_loss.item())

        if writer and batch_idx % config['logging']['log_interval'] == 0:
            current_iter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/cls_loss', cls_loss.item(), current_iter)
            writer.add_scalar('train/reg_loss', reg_loss.item(), current_iter)
            writer.add_scalar('train/seg_loss', seg_loss.item(), current_iter)
            writer.add_scalar('train/total_loss', loss.item(), current_iter)
    return total_loss / len(train_loader) if len(train_loader) > 0 else 0

def validate(backbone, det_depth_model, seg_model, val_loader, criterion, device, epoch, writer, wseg):
    backbone.eval()
    det_depth_model.eval()
    seg_model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets_cls = targets['cls'].to(device)
            targets_box_depth = targets['box_and_depth'].to(device)
            targets_seg = targets['segmentation'].to(device)

            res3_reduced_feat, res4_reduced_feat, res5_reduced_feat = backbone(images)
            
            # DSPNet_Detector now expects res3_reduced (128ch) and res4_reduced (256ch)
            det_cls_pred, det_box_depth_pred = det_depth_model(res3_reduced_feat, res4_reduced_feat)
            
            # DSPNetSegmentationHead expects res3_reduced, res4_reduced, res5_reduced
            seg_out = seg_model(res3_reduced_feat, res4_reduced_feat, res5_reduced_feat)
            
            cls_loss = criterion['cls'](det_cls_pred.view(-1, det_cls_pred.size(-1)), targets_cls.view(-1))
            reg_loss = criterion['reg'](det_box_depth_pred, targets_box_depth)
            seg_loss = criterion['segmentation'](seg_out, targets_seg)
            loss = cls_loss + reg_loss + wseg * seg_loss
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    if writer:
        writer.add_scalar('val/total_loss', avg_loss, epoch)
    return avg_loss

def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    backbone = ResNetBackbone(pretrained=config['model']['pretrained']).to(device)
    
    num_det_classes = config['model'].get('num_detection_classes', 20) 
    anchors_per_loc = config['model'].get('anchors_per_location_list', [4, 6, 6, 6, 4, 4])
    if len(anchors_per_loc) != 6:
        raise ValueError("config.model.anchors_per_location_list must have 6 elements.")

    detection_model = DSPNet_Detector(
        num_classes=num_det_classes, 
        anchors_per_location_list=anchors_per_loc
    ).to(device)
    
    segmentation_model = DSPNetSegmentationHead(
        num_classes=config['model']['num_classes']
    ).to(device)
    
    train_loader, val_loader = create_dataloaders(config)
    if not train_loader or not val_loader:
        print("Dataloaders are not initialized. Exiting. Please implement create_dataloaders.")
        return

    criterion = {
        'cls': nn.CrossEntropyLoss(), 
        'reg': nn.SmoothL1Loss(), 
        'segmentation': nn.CrossEntropyLoss()
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
    best_val_loss = float('inf') # Initialize best_val_loss here

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
            # best_val_loss = float('inf') # Already initialized
    # else:
        # best_val_loss = float('inf') # Already initialized
    
    wseg = config['training'].get('segmentation_loss_weight', 4.0) 

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f'Epoch {epoch+1}/{config['training']['epochs']}')
        
        train_loss = train(backbone, detection_model, segmentation_model, train_loader, optimizer, criterion, device, epoch, writer, config, wseg)
        print(f'Train Loss: {train_loss:.4f}')
        
        val_loss = validate(backbone, detection_model, segmentation_model, val_loader, criterion, device, epoch, writer, wseg)
        print(f'Validation Loss: {val_loss:.4f}')
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(output_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'detection_model_state_dict': detection_model.state_dict(),
                'segmentation_model_state_dict': segmentation_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, save_path)
            print(f"Saved best model to {save_path}")
        
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            save_path_interval = os.path.join(output_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'detection_model_state_dict': detection_model.state_dict(),
                'segmentation_model_state_dict': segmentation_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss, 
                 'config': config
            }, save_path_interval)
            print(f"Saved model checkpoint to {save_path_interval}")

    if writer:
        writer.close()
    print("Training completed.")

if __name__ == '__main__':
    main() 