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
        self.layer1 = resnet.layer1  # Original C2 output (256 channels), not directly used by current heads
        self.layer2 = resnet.layer2  # Original C3 output (512 channels)
        self.layer3 = resnet.layer3  # Original C4 output (1024 channels)
        self.layer4 = resnet.layer4  # Original C5 output (2048 channels)

        # Channel reduction layers are removed

    def forward(self, x):
        x = self.stem(x)
        _ = self.layer1(x)    # C2 output, for ResNet flow
        c3_original = self.layer2(x)     # Original C3 (H/8, 512ch)
        c4_original = self.layer3(c3_original)  # Original C4 (H/16, 1024ch)
        c5_original = self.layer4(c4_original)  # Original C5 (H/32, 2048ch)

        # Return the original feature maps
        return c3_original, c4_original, c5_original 