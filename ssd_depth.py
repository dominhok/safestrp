import torch
import torch.nn as nn
# torchvision.models is not used here anymore as backbone is external

# 1. Modified conv_unit
def conv_unit(in_channels, mid_channels, out_channels, stride_3x3=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=stride_3x3),
        nn.ReLU(inplace=True)
    )

# 2. ResNetBackbone Implementation
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        resnet = models.resnet50(weights=weights)

        # Remove fully connected layer and average pooling
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # conv2_x
        self.layer2 = resnet.layer2 # conv3_x
        self.layer3 = resnet.layer3 # conv4_x -> res3_x for DSPNet (1024 channels)
        self.layer4 = resnet.layer4 # conv5_x -> res4_x for DSPNet (2048 channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        res3_x_feat = self.layer3(x)       # Output of ResNet layer3 (e.g., H/16, W/16, 1024 channels)
        res4_x_feat = self.layer4(res3_x_feat) # Output of ResNet layer4 (e.g., H/32, W/32, 2048 channels)
        return res3_x_feat, res4_x_feat

# SSDDecoder class is removed/commented out as its logic is integrated into DSPNet_Detector
# class SSDDecoder(nn.Module):
#     ... (previous content)

# 3. PredictionHead (No changes needed here, kept for context)
class PredictionHead(nn.Module):
    def __init__(self, in_channels, num_anchors_per_location, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location
        self.classifier_head = nn.Conv2d(in_channels,
                                         num_anchors_per_location * (num_classes + 1),
                                         kernel_size=3, padding=1)
        self.regressor_head = nn.Conv2d(in_channels,
                                        num_anchors_per_location * (4 + 1), # 4 for bbox, 1 for depth
                                        kernel_size=3, padding=1)

    def forward(self, feature_map):
        cls_preds = self.classifier_head(feature_map)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.size(0), -1, self.num_classes + 1)
        reg_preds = self.regressor_head(feature_map)
        reg_preds = reg_preds.permute(0, 2, 3, 1).contiguous()
        reg_preds = reg_preds.view(reg_preds.size(0), -1, 5)
        return cls_preds, reg_preds

# 4. DSPNet_Detector (Modified to use channel-reduced backbone outputs and align with diagram)
class DSPNet_Detector(nn.Module):
    def __init__(self, num_classes, anchors_per_location_list): 
        super().__init__()
        if len(anchors_per_location_list) != 6:
            # Expecting 6 prediction sources: res4_reduced, res5_reduced, and 4 extra layers
            raise ValueError("anchors_per_location_list must have 6 elements for 6 feature sources.")

        # Input channels from ResNetBackbone (reduced features)
        self.input_res3_reduced_channels = 128 # C3_reduced from backbone
        self.input_res4_reduced_channels = 256 # C4_reduced from backbone

        # Extra Layers: Start from res4_reduced_feat, 4 layers in total
        # Output channels and structure inspired by DSPNet diagram's featX_2 layers, but starting from res4_reduced

        # el1: Input from res4_reduced (256ch). Target output similar to feat1_2 (512ch)
        el1_out_channels = 512 
        self.extra_layer1 = conv_unit(self.input_res4_reduced_channels, mid_channels=256, out_channels=el1_out_channels, stride_3x3=2)
        
        # el2: Input from extra_layer1 (512ch). Target output similar to feat2_2 (256ch)
        el2_out_channels = 256
        self.extra_layer2 = conv_unit(el1_out_channels, mid_channels=128, out_channels=el2_out_channels, stride_3x3=2)
        
        # el3: Input from extra_layer2 (256ch). Target output similar to feat3_2 (128ch)
        el3_out_channels = 128
        self.extra_layer3 = conv_unit(el2_out_channels, mid_channels=64, out_channels=el3_out_channels, stride_3x3=2)
        
        # el4: Input from extra_layer3 (128ch). Target output similar to feat4_2 (64ch)
        el4_out_channels = 64
        self.extra_layer4 = conv_unit(el3_out_channels, mid_channels=32, out_channels=el4_out_channels, stride_3x3=2) 

        self.pred_heads = nn.ModuleList([
            PredictionHead(self.input_res3_reduced_channels, anchors_per_location_list[0], num_classes), # Source 0: res3_reduced
            PredictionHead(self.input_res4_reduced_channels, anchors_per_location_list[1], num_classes), # Source 1: res4_reduced
            PredictionHead(el1_out_channels, anchors_per_location_list[2], num_classes),                 # Source 2: extra_layer1 output
            PredictionHead(el2_out_channels, anchors_per_location_list[3], num_classes),                 # Source 3: extra_layer2 output
            PredictionHead(el3_out_channels, anchors_per_location_list[4], num_classes),                 # Source 4: extra_layer3 output
            PredictionHead(el4_out_channels, anchors_per_location_list[5], num_classes)                  # Source 5: extra_layer4 output
        ])

    # Accepts res3_reduced_feat and res4_reduced_feat from an external backbone
    def forward(self, res3_reduced_feat, res4_reduced_feat):
        # res3_reduced_feat: (Batch, 128, H/8, W/8)
        # res4_reduced_feat: (Batch, 256, H/16, W/16)

        # Extra layer features are derived from res4_reduced_feat
        feat_source_el1 = self.extra_layer1(res4_reduced_feat) 
        feat_source_el2 = self.extra_layer2(feat_source_el1)   
        feat_source_el3 = self.extra_layer3(feat_source_el2)   
        feat_source_el4 = self.extra_layer4(feat_source_el3)

        feature_sources_for_heads = [
            res3_reduced_feat, # Source 0
            res4_reduced_feat, # Source 1
            feat_source_el1,   # Source 2
            feat_source_el2,   # Source 3
            feat_source_el3,   # Source 4
            feat_source_el4    # Source 5
        ]

        all_cls_preds = []
        all_reg_preds = []

        for i, feature_map_input in enumerate(feature_sources_for_heads):
            cls_preds, reg_preds = self.pred_heads[i](feature_map_input)
            all_cls_preds.append(cls_preds)
            all_reg_preds.append(reg_preds)

        final_cls_preds = torch.cat(all_cls_preds, dim=1)
        final_reg_preds = torch.cat(all_reg_preds, dim=1)

        return final_cls_preds, final_reg_preds

# Example usage (for testing the structure - needs to be adapted if run standalone)
# if __name__ == '__main__':
#     # Assuming resnet_backbone.py returns res3_reduced, res4_reduced, res5_reduced
#     # from projects.safestrp.resnet_backbone import ResNetBackbone 
#     # backbone_test = ResNetBackbone(pretrained=False)
#     # dummy_image_input = torch.randn(1, 3, 512, 1024) 
#     
#     # res3_r_test, res4_r_test, _ = backbone_test(dummy_image_input) # We need res3_reduced and res4_reduced
#     # print(f"res3_reduced shape: {res3_r_test.shape}") # Expected: B, 128, H/8, W/8
#     # print(f"res4_reduced shape: {res4_r_test.shape}") # Expected: B, 256, H/16, W/16
#
#     num_classes_example = 20 
#     anchors_per_loc_list_example = [4, 6, 6, 6, 4, 4] 
#     
#     dspnet_detector_head = DSPNet_Detector(
#         num_classes=num_classes_example,
#         anchors_per_location_list=anchors_per_loc_list_example
#     )
#     dspnet_detector_head.eval()
# 
#     # Create dummy reduced C3 and C4 feature maps (as if from backbone)
#     # Input H=512, W=1024
#     # res3_reduced (H/8, W/8): 512/8=64, 1024/8=128. Shape: (1, 128, 64, 128)
#     # res4_reduced (H/16, W/16): 512/16=32, 1024/16=64. Shape: (1, 256, 32, 64)
#     dummy_res3_reduced_feat = torch.randn(1, 128, 64, 128)
#     dummy_res4_reduced_feat = torch.randn(1, 256, 32, 64)
# 
#     with torch.no_grad():
#         cls_predictions, reg_predictions = dspnet_detector_head(dummy_res3_reduced_feat, dummy_res4_reduced_feat)
# 
#     print("DSPNet_Detector (Head Only with res3_reduced, res4_reduced inputs) Test:")
#     print("Classification Predictions Shape:", cls_predictions.shape)
#     print("Regression (bbox + depth) Predictions Shape:", reg_predictions.shape)
# 
#     # Calculate expected total anchors for H=512, W=1024 input
#     # Source 0 (res3_reduced, 64x128): (64*128) * anchors_per_loc_list_example[0]
#     # Source 1 (res4_reduced, 32x64): (32*64) * anchors_per_loc_list_example[1]
#     # Source 2 (el1 from res4_reduced, stride=2, 16x32): (16*32) * anchors_per_loc_list_example[2]
#     # Source 3 (el2 from el1, stride=2, 8x16): (8*16) * anchors_per_loc_list_example[3]
#     # Source 4 (el3 from el2, stride=2, 4x8): (4*8) * anchors_per_loc_list_example[4]
#     # Source 5 (el4 from el3, stride=2, 2x4): (2*4) * anchors_per_loc_list_example[5]
#     
#     # L0 (res3_r): 8192 * 4 = 32768
#     # L1 (res4_r): 2048 * 6 = 12288
#     # L2 (el1):    512 * 6 = 3072
#     # L3 (el2):    128 * 6 = 768
#     # L4 (el3):     32 * 4 = 128
#     # L5 (el4):      8 * 4 = 32
#     # Total (example for [4,6,6,6,4,4]): 32768 + 12288 + 3072 + 768 + 128 + 32 = 49056
#     # print(f"Expected total anchors for H=512, W=1024: {49056}")
