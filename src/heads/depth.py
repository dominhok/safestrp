"""
Depth regression head for depth estimation task.

Contains DepthRegressionHead for dense depth prediction with skip connections.
"""

import torch
import torch.nn as nn

from .base import BaseHead


class DepthRegressionHead(BaseHead):
    """
    Depth regression head with skip connections for better detail preservation.
    
    Uses multi-scale features from backbone and applies skip connections
    to preserve fine-grained depth information.
    """
    
    def __init__(self, 
                 c3_channels: int = 512,
                 c4_channels: int = 1024, 
                 c5_channels: int = 2048,
                 output_channels: int = 1):
        super().__init__()
        
        # Main upsampling path from C5
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c5_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Skip connection from C4
        self.skip_c4 = nn.Sequential(
            nn.Conv2d(c4_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Combined upsampling
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),  # 512+512 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Skip connection from C3
        self.skip_c3 = nn.Sequential(
            nn.Conv2d(c3_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling to original resolution
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # 256+256 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final depth prediction
        self.depth_pred = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._init_weights()
        
        print(f"✅ DepthRegressionHead with Skip Connections:")
        print(f"   Input: C3({c3_channels}), C4({c4_channels}), C5({c5_channels})")
        print(f"   Output: {output_channels} channels")
        print(f"   Skip connections: C4 -> Up2, C3 -> Up3")
    
    def forward(self, c3_feat: torch.Tensor, c4_feat: torch.Tensor, c5_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections.
        
        Args:
            c3_feat: (B, 512, H/8, W/8)
            c4_feat: (B, 1024, H/16, W/16) 
            c5_feat: (B, 2048, H/32, W/32)
            
        Returns:
            depth: (B, 1, H, W)
        """
        # Upsample C5 to C4 resolution
        up1 = self.up1(c5_feat)  # (B, 512, H/16, W/16)
        
        # Skip connection from C4
        skip_c4 = self.skip_c4(c4_feat)  # (B, 512, H/16, W/16)
        
        # Combine and upsample to C3 resolution
        combined1 = torch.cat([up1, skip_c4], dim=1)  # (B, 1024, H/16, W/16)
        up2 = self.up2(combined1)  # (B, 256, H/8, W/8)
        
        # Skip connection from C3
        skip_c3 = self.skip_c3(c3_feat)  # (B, 256, H/8, W/8)
        
        # Combine and upsample to original resolution
        combined2 = torch.cat([up2, skip_c3], dim=1)  # (B, 512, H/8, W/8)
        up3 = self.up3(combined2)  # (B, 32, H, W)
        
        # Final depth prediction
        depth = self.depth_pred(up3)  # (B, 1, H, W)
        
        return depth 