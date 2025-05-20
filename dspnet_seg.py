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

        # Input channels from ResNetBackbone (original channels)
        self.res3_in_channels = 512   # C3_original
        self.res4_in_channels = 1024  # C4_original
        self.res5_in_channels = 2048  # C5_original

        # Process C3 path
        self.path_c3 = nn.Sequential(
            nn.Conv2d(self.res3_in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Process C4 path
        self.path_c4 = nn.Sequential(
            nn.Conv2d(self.res4_in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Process C5 path for Pyramid Pooling Module (PPM)
        self.path_c5_for_ppm = nn.Sequential(
            nn.Conv2d(self.res5_in_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        ppm_in_channels = 512 
        self.ppm_pool_scales = [1, 2, 3, 6] 
        ppm_inter_channels = ppm_in_channels // len(self.ppm_pool_scales)

        self.ppm_convs = nn.ModuleList()
        for scale in self.ppm_pool_scales:
            self.ppm_convs.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(scale, scale)),
                nn.Conv2d(ppm_in_channels, ppm_inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(ppm_inter_channels),
                nn.ReLU(inplace=True)
            ))
        
        ppm_concat_channels = ppm_in_channels + len(self.ppm_pool_scales) * ppm_inter_channels # 512 (original path) + 4 * 128 = 1024
        
        fuse_in_channels = 128 + 256 + ppm_concat_channels # 128 (from C3) + 256 (from C4) + 1024 (from PPM) = 1408
        
        self.fuse_conv1 = nn.Conv2d(fuse_in_channels, 256, kernel_size=3, padding=1, bias=False)
        self.fuse_bn1 = nn.BatchNorm2d(256)
        self.fuse_relu1 = nn.ReLU(inplace=True)
        self.fuse_conv2 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

        self.deconv = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        if self.deconv.weight.requires_grad:
             bilinear_kernel_init(self.deconv.weight)

    def forward(self, c3_feat, c4_feat, c5_feat):
        target_h_8, target_w_8 = c3_feat.shape[2:]
        target_size_h8_w8 = (target_h_8, target_w_8)

        path_c3_out = self.path_c3(c3_feat)
        
        path_c4_processed = self.path_c4(c4_feat)
        path_c4_out = F.interpolate(path_c4_processed, size=target_size_h8_w8, mode='bilinear', align_corners=False)

        c5_ppm_input = self.path_c5_for_ppm(c5_feat)
        
        ppm_branch_outputs = []
        for ppm_conv_layer in self.ppm_convs:
            pooled = ppm_conv_layer(c5_ppm_input)
            ppm_branch_outputs.append(F.interpolate(pooled, size=c5_ppm_input.shape[2:], mode='bilinear', align_corners=False))
        
        # Concatenate the original feature map (c5_ppm_input) with the outputs of the pooling branches
        ppm_concat = torch.cat([c5_ppm_input] + ppm_branch_outputs, dim=1)
        ppm_fused_upsampled = F.interpolate(ppm_concat, size=target_size_h8_w8, mode='bilinear', align_corners=False)

        concat_features = torch.cat([path_c3_out, path_c4_out, ppm_fused_upsampled], dim=1)

        fused_out = self.fuse_relu1(self.fuse_bn1(self.fuse_conv1(concat_features)))
        final_logits = self.fuse_conv2(fused_out)

        final_seg_out = self.deconv(final_logits)
        return final_seg_out
