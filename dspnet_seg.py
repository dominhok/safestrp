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

        # Conv on res3 and res4
        self.res3_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res4_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        # Global prior from res5 with multi-scale pooling
        self.pool_scales = [1, 2, 4]
        self.global_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(512, out_ch, kernel_size=1, bias=False)
            ) for scale, out_ch in zip(self.pool_scales, [512, 256, 128])
        ])

        # Fuse all features (128+256+512+256+128 = 1280)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        # Final upsampling by 4x using learnable deconv
        self.deconv = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=4, padding=2, bias=False)
        bilinear_kernel_init(self.deconv.weight)

    def forward(self, res3, res4, res5):
        # Prepare local features
        size = res3.shape[2:]  # H/8, W/8
        res3_feat = self.res3_conv(res3)                    # (B, 128, H/8, W/8)
        res4_feat = F.interpolate(self.res4_conv(res4), size=size, mode='bilinear', align_corners=False)  # (B, 256, H/8, W/8)

        # Global prior from res5
        global_feats = []
        for layer in self.global_convs:
            x = layer(res5)  # (B, C, scale, scale)
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            global_feats.append(x)

        # Concatenate all features
        concat = torch.cat([res3_feat, res4_feat] + global_feats, dim=1)  # (B, 1280, H/8, W/8)

        # Fuse and predict
        out = self.fuse_conv(concat)  # (B, num_classes, H/8, W/8)

        # Final upsample to H/2 resolution (×4)
        out = self.deconv(out)  # → (B, num_classes, H/2, W/2)
        return out
