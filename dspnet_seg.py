import torch
import torch.nn as nn
import torch.nn.functional as F

def bilinear_kernel_init(w):
    """Initialize transposed conv kernel with bilinear upsampling weights"""
    c1, c2, k, _ = w.shape #weight shape: (out_channels, in_channels, kernel_size, kernel_size)
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

        # Local feature conv
        self.res3_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res4_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        # Global prior: from res5
        self.global_pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size),
                nn.Conv2d(512, out_ch, kernel_size=1, bias=False)
            ) for output_size, out_ch in zip([(4, 8), (8, 16), (16, 32)], [128, 256, 512]) # 3 avg pool 
        ])

        # Final fusion conv (total input channels = 1792)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(1792, 256, kernel_size=3, padding=1, bias=False), #bias=False, SegNet 기반
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        # Upsample to H/4 ("decrease the segmentation output size by 4-fold in terms of both height and width")
        self.deconv = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        if self.deconv.weight.requires_grad:
            bilinear_kernel_init(self.deconv.weight)

    def forward(self, res3_feat, res4_feat, res5_feat):
        res3_out = self.res3_conv(res3_feat)  # (B,128,H/8,W/8)
        res4_out = F.interpolate(self.res4_conv(res4_feat), size=res3_out.shape[2:], mode='bilinear', align_corners=False)  # (B,256,H/8,W/8)

        # Global path
        global_feats = []
        global_feats.append(F.interpolate(res5_feat, size=res3_out.shape[2:], mode='bilinear', align_corners=False))  # (B,512,H/8,W/8)
        for pool in self.global_pools:
            pooled = pool(res5_feat)  # (B,C,H',W') depending on scale
            upsampled = F.interpolate(pooled, size=res3_out.shape[2:], mode='bilinear', align_corners=False)
            global_feats.append(upsampled)

        # Concat all
        concat = torch.cat([res3_out, res4_out] + global_feats, dim=1)  # (B,1792,H/8,W/8)

        out = self.fuse_conv(concat)  # (B,num_classes,H/8,W/8)
        out = self.deconv(out)       # (B,num_classes,H/4,W/4)
        return out
