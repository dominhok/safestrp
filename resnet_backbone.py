import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        # Initial layers
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        # Residual blocks (original outputs)
        self.layer1 = resnet.layer1  # Original C2 output (256 channels)
        self.layer2 = resnet.layer2  # Original C3 output (512 channels)
        self.layer3 = resnet.layer3  # Original C4 output (1024 channels)
        self.layer4 = resnet.layer4  # Original C5 output (2048 channels)

        # 1x1 Convolutions with BatchNorm and ReLU to reduce channels for DSPNet heads
        self.c3_reduced_channels = 128
        self.reduce_c3 = nn.Sequential(
            nn.Conv2d(512, self.c3_reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c3_reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        self.c4_reduced_channels = 256
        self.reduce_c4 = nn.Sequential(
            nn.Conv2d(1024, self.c4_reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c4_reduced_channels),
            nn.ReLU(inplace=True)
        )

        self.c5_reduced_channels = 512
        self.reduce_c5 = nn.Sequential(
            nn.Conv2d(2048, self.c5_reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c5_reduced_channels),
            nn.ReLU(inplace=True)
        )

        # Initialize weights for the new 1x1 conv and BN layers (optional but good practice)
        for m in [self.reduce_c3, self.reduce_c4, self.reduce_c5]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(layer, nn.BatchNorm2d):
                        nn.init.constant_(layer.weight, 1)
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        _ = self.layer1(x)    # C2 output, not directly used by heads in diagram but part of ResNet flow
        c3_original = self.layer2(x)     # Original C3 (H/8, 512ch)
        c4_original = self.layer3(c3_original)  # Original C4 (H/16, 1024ch)
        c5_original = self.layer4(c4_original)  # Original C5 (H/32, 2048ch)

        # Apply 1x1 convolutions (with BN and ReLU) to get reduced feature maps
        res3_reduced = self.reduce_c3(c3_original) # (H/8, 128ch)
        res4_reduced = self.reduce_c4(c4_original) # (H/16, 256ch)
        res5_reduced = self.reduce_c5(c5_original) # (H/32, 512ch)
        
        # Return the reduced feature maps as per diagram's input requirements for heads
        return res3_reduced, res4_reduced, res5_reduced 