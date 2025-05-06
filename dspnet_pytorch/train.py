import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import argparse
import yaml
from tqdm import tqdm
import os
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR

# Import model
import detAndDepthModel
import segModel
from resnet_backbone import ResNetBackbone

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-task Learning Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
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
    train_loader = None
    val_loader = None
    return train_loader, val_loader

def train(backbone, det_depth_model, seg_model, train_loader, optimizer, criterion, device, epoch, writer, config, wseg):
    backbone.train()
    det_depth_model.train()
    seg_model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        # Forward pass through backbone
        res3, res4, res5 = backbone(images)
        # Detection and depth predictions
        det_pred, box_and_depth_pred = det_depth_model(res3, res4, res5)
        # Segmentation
        seg_out = seg_model(res3, res4, res5)  # Segmentation model receives all features
        
        # Compute losses in the training loop
        cls_loss = criterion['cls'](det_pred, targets['cls'])
        reg_loss = criterion['reg'](box_and_depth_pred, targets['box_and_depth'])
        seg_loss = criterion['segmentation'](seg_out, targets['segmentation'])

        # Multi-task loss
        loss = cls_loss + reg_loss + wseg * seg_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Log to tensorboard
        if batch_idx % config['logging']['log_interval'] == 0:
            writer.add_scalar('train/cls_loss', cls_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/reg_loss', reg_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/seg_loss', seg_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/total_loss', loss.item(), epoch * len(train_loader) + batch_idx)
    return total_loss / len(train_loader)

def validate(backbone, det_depth_model, seg_model, val_loader, criterion, device, epoch, writer, wseg):
    backbone.eval()
    det_depth_model.eval()
    seg_model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            res3, res4, res5 = backbone(images)
            det_pred, box_and_depth_pred = det_depth_model(res3, res4, res5)
            seg_out = seg_model(res3, res4, res5)
            cls_loss = criterion['cls'](det_pred, targets['cls'])
            reg_loss = criterion['reg'](box_and_depth_pred, targets['box_and_depth'])
            seg_loss = criterion['segmentation'](seg_out, targets['segmentation'])
            loss = cls_loss + reg_loss + wseg * seg_loss
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    writer.add_scalar('val/total_loss', avg_loss, epoch)
    return avg_loss

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained ResNet-50 backbone
    backbone = ResNetBackbone(pretrained=True).to(device)
    
    # Initialize detection and segmentation models
    detection_model = detAndDepthModel.DetAndDepthHead(config['model']).to(device)
    segmentation_model = segModel.SegmentationHead(config['model']).to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Define loss functions
    criterion = {
        'cls': nn.CrossEntropyLoss(),        # For detection classification
        'reg': nn.L1Loss(),                 # For box and depth regression
        'segmentation': nn.CrossEntropyLoss() # For semantic segmentation
    }
    
    # Combine parameters for optimizer
    params = list(backbone.parameters()) + list(detection_model.parameters()) + list(segmentation_model.parameters())
    optimizer = optim.SGD(params, lr=0.0005, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[80, 160, 240], gamma=0.5)
    
    # Create tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{timestamp}')
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        detection_model.load_state_dict(checkpoint['detection_model_state_dict'])
        segmentation_model.load_state_dict(checkpoint['segmentation_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    best_val_loss = float('inf')
    wseg = 4  # Segmentation loss weight
    for epoch in range(start_epoch, 320):
        print(f'Epoch {epoch+1}/320')
        
        # Train
        train_loss = train(backbone, detection_model, segmentation_model, train_loader, optimizer, criterion, device, epoch, writer, config, wseg)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = validate(backbone, detection_model, segmentation_model, val_loader, criterion, device, epoch, writer, wseg)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'detection_model_state_dict': detection_model.state_dict(),
                'segmentation_model_state_dict': segmentation_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'checkpoints/best_model_{timestamp}.pth')
        
        scheduler.step()
    
    writer.close()

if __name__ == '__main__':
    main() 