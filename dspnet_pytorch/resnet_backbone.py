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
        # Residual blocks
        self.layer1 = resnet.layer1  # res2
        self.layer2 = resnet.layer2  # res3
        self.layer3 = resnet.layer3  # res4
        self.layer4 = resnet.layer4  # res5

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)        # res2
        res3 = self.layer2(x)     # res3
        res4 = self.layer3(res3)  # res4
        res5 = self.layer4(res4)  # res5
        return res3, res4, res5 