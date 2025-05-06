import torch
import torch.nn as nn
import torch.nn.functional as F

def bilinear_kernel_init(w):
    """Initialize transposed conv kernel with bilinear upsampling weights"""
    c1, c2, k, _ = w.shape
    f = (k + 1) // 2
    center = f - 1 if k % 2 == 1 else f - 0.5
    og = torch.arange(k).float()
    filt = (1 - torch.abs(og - center) / f).unsqueeze(0)
    weight = filt.T @ filt
    w.data.zero_()
    for i in range(c1):
        for j in range(c2):
            if i == j:
                w.data[i, j] = weight

class DSPNetSegmentationHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Input channels from ResNetBackbone (reduced features)
        self.res3_reduced_in_channels = 128 # From backbone.reduce_c3
        self.res4_reduced_in_channels = 256 # From backbone.reduce_c4
        self.res5_reduced_in_channels = 512 # From backbone.reduce_c5

        # Output channels for these intermediate convs, aligned with diagram
        res3_conv_out_channels = 64    # Diagram suggests 64ch from res3_reduced path
        res4_conv_out_channels = 128   # Diagram suggests 128ch from res4_reduced path

        # Conv on res3_reduced and res4_reduced
        self.res3_conv = nn.Conv2d(self.res3_reduced_in_channels, res3_conv_out_channels, kernel_size=3, padding=1, bias=False)
        self.res4_conv = nn.Conv2d(self.res4_reduced_in_channels, res4_conv_out_channels, kernel_size=3, padding=1, bias=False)

        # Global prior from res5_reduced with multi-scale pooling (aligned with diagram: 4 scales)
        self.pool_scales = [1, 2, 4, 8] # Output spatial sizes for AdaptiveAvgPool2d
        # Output channels after 1x1 conv on pooled features, aligned with diagram
        global_pool_out_channels = [512, 256, 128, 64] 
        
        self.global_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(self.res5_reduced_in_channels, out_ch, kernel_size=1, bias=False)
            ) for scale, out_ch in zip(self.pool_scales, global_pool_out_channels)
        ])

        # Calculate in_channels for fuse_conv, aligned with diagram
        # Diagram: 64 (from res3 path) + 128 (from res4 path) + (512+256+128+64 from global_pools) = 1152
        fuse_in_channels = res3_conv_out_channels + res4_conv_out_channels + sum(global_pool_out_channels)
        # Expected: 64 + 128 + 960 = 1152

        # Fuse all features
        # The output channel of the first conv in fuse_conv (e.g., 256) can be tuned.
        # Diagram does not specify this intermediate channel count for the fuse block.
        # Let's keep it at 256 for now, as in the original user code for fuse_conv.
        fuse_intermediate_channels = 256 
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fuse_in_channels, fuse_intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_intermediate_channels, num_classes, kernel_size=1, bias=False)
        )

        self.deconv = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=4, padding=2, bias=False)
        if self.deconv.weight.requires_grad:
             bilinear_kernel_init(self.deconv.weight)

    # Arguments renamed to reflect they are channel-reduced features from the backbone
    def forward(self, res3_reduced_feat, res4_reduced_feat, res5_reduced_feat):
        # res3_reduced_feat: (B, 128, H/8, W/8)
        # res4_reduced_feat: (B, 256, H/16, W/16)
        # res5_reduced_feat: (B, 512, H/32, W/32)

        target_size = res3_reduced_feat.shape[2:] # Target spatial size is H/8, W/8
        
        # Process res3_reduced_feat path
        # Output: (B, 64, H/8, W/8)
        path_res3_out = self.res3_conv(res3_reduced_feat)
        
        # Process res4_reduced_feat path
        # Input: (B, 256, H/16, W/16), Output of conv: (B, 128, H/16, W/16)
        path_res4_processed = self.res4_conv(res4_reduced_feat)
        # Upsample to target_size (H/8, W/8). Output: (B, 128, H/8, W/8)
        path_res4_out = F.interpolate(path_res4_processed, size=target_size, mode='bilinear', align_corners=False)

        # Global prior from res5_reduced_feat
        global_feats_list = []
        for layer in self.global_convs:
            x = layer(res5_reduced_feat) # Pooled and 1x1 convolved
            # Upsample to target_size (H/8, W/8)
            x_upsampled = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            global_feats_list.append(x_upsampled)

        # Concatenate all features (all should be at H/8, W/8 spatial resolution)
        # Channels: 64 (res3 path) + 128 (res4 path) + 960 (global sum) = 1152
        concat_features = torch.cat([path_res3_out, path_res4_out] + global_feats_list, dim=1)

        # Fuse and predict
        # Input: (B, 1152, H/8, W/8), Output: (B, num_classes, H/8, W/8)
        fused_out = self.fuse_conv(concat_features)

        # Final upsample to H/2 resolution (×4 from H/8)
        final_seg_out = self.deconv(fused_out) # -> (B, num_classes, H/2, W/2)
        return final_seg_out
