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
                                         num_anchors_per_location * (num_classes + 1), # +1 for background
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

# 4. DSPNet_Detector (Modified to align with dspnet's ResNet-50 feature sources)
class DSPNet_Detector(nn.Module):
    def __init__(self, num_classes, anchors_per_location_list, min_filter_extra_layers=128):
        super().__init__()
        if len(anchors_per_location_list) != 7: # 3 backbone features + 4 extra layers
            raise ValueError("anchors_per_location_list must have 7 elements for 7 feature sources.")

        # Input channels from ResNet50 Backbone (original channels, no reduction here)
        # These are typical output channels for ResNet-50 stages
        self.input_res3_channels = 512   # C3 output (e.g., resnet_backbone.layer2 output)
        self.input_res4_channels = 1024  # C4 output (e.g., resnet_backbone.layer3 output)
        self.input_res5_channels = 2048  # C5 output (e.g., resnet_backbone.layer4 output)

        # Extra Layers: Start from res5_feat (C5), 4 layers in total
        # Output channels for extra layers based on dspnet's num_filters = [..., 512, 256, 256, 128]
        
        # el1: Input from res5_feat (2048ch). Target output 512ch. Stride 2.
        el1_out_channels = 512
        el1_mid_channels = max(min_filter_extra_layers, el1_out_channels // 2)
        self.extra_layer1 = conv_unit(self.input_res5_channels, mid_channels=el1_mid_channels, out_channels=el1_out_channels, stride_3x3=2)

        # el2: Input from extra_layer1 (512ch). Target output 256ch. Stride 2.
        el2_out_channels = 256
        el2_mid_channels = max(min_filter_extra_layers, el2_out_channels // 2)
        self.extra_layer2 = conv_unit(el1_out_channels, mid_channels=el2_mid_channels, out_channels=el2_out_channels, stride_3x3=2)

        # el3: Input from extra_layer2 (256ch). Target output 256ch. Stride 2.
        el3_out_channels = 256
        el3_mid_channels = max(min_filter_extra_layers, el3_out_channels // 2)
        self.extra_layer3 = conv_unit(el2_out_channels, mid_channels=el3_mid_channels, out_channels=el3_out_channels, stride_3x3=2)

        # el4: Input from extra_layer3 (256ch). Target output 128ch. Stride 2.
        el4_out_channels = 128
        # Corrected mid_channels based on dspnet common.py: max(min_filter, out_channels // 2)
        el4_mid_channels = max(min_filter_extra_layers, el4_out_channels // 2) # max(128, 128//2) = max(128, 64) = 128
        self.extra_layer4 = conv_unit(el3_out_channels, mid_channels=el4_mid_channels,  out_channels=el4_out_channels, stride_3x3=2)

        # Prediction heads for each feature source
        # The input channels to PredictionHead must match the output channels of the feature source
        self.pred_heads = nn.ModuleList([
            PredictionHead(self.input_res3_channels, anchors_per_location_list[0], num_classes), # Source 0: C3
            PredictionHead(self.input_res4_channels, anchors_per_location_list[1], num_classes), # Source 1: C4
            PredictionHead(self.input_res5_channels, anchors_per_location_list[2], num_classes), # Source 2: C5
            PredictionHead(el1_out_channels, anchors_per_location_list[3], num_classes),         # Source 3: extra_layer1 output
            PredictionHead(el2_out_channels, anchors_per_location_list[4], num_classes),         # Source 4: extra_layer2 output
            PredictionHead(el3_out_channels, anchors_per_location_list[5], num_classes),         # Source 5: extra_layer3 output
            PredictionHead(el4_out_channels, anchors_per_location_list[6], num_classes)          # Source 6: extra_layer4 output
        ])

    # Accepts res3_feat, res4_feat, res5_feat from an external ResNet-50 backbone
    def forward(self, res3_feat, res4_feat, res5_feat):
        # Expected shapes assuming input H, W (e.g., 512, 1024 for dspnet)
        # res3_feat: (Batch, 512, H/8, W/8)
        # res4_feat: (Batch, 1024, H/16, W/16)
        # res5_feat: (Batch, 2048, H/32, W/32)

        # Extra layer features are derived from res5_feat
        feat_source_el1 = self.extra_layer1(res5_feat)      # Output: (Batch, 512, H/64, W/64)
        feat_source_el2 = self.extra_layer2(feat_source_el1)  # Output: (Batch, 256, H/128, W/128)
        feat_source_el3 = self.extra_layer3(feat_source_el2)  # Output: (Batch, 256, H/256, W/256)
        feat_source_el4 = self.extra_layer4(feat_source_el3)  # Output: (Batch, 128, H/512, W/512)

        feature_sources_for_heads = [
            res3_feat,         # Source 0
            res4_feat,         # Source 1
            res5_feat,         # Source 2
            feat_source_el1,   # Source 3
            feat_source_el2,   # Source 4
            feat_source_el3,   # Source 5
            feat_source_el4    # Source 6
        ]

        all_cls_preds = []
        all_reg_preds = []

        for i, feature_map_input in enumerate(feature_sources_for_heads):
            cls_preds, reg_preds = self.pred_heads[i](feature_map_input)
            all_cls_preds.append(cls_preds)
            all_reg_preds.append(reg_preds)

        # Concatenate predictions from all feature sources
        # Output shape: (Batch, TotalNumberOfAnchors, NumClasses + 1)
        final_cls_preds = torch.cat(all_cls_preds, dim=1)
        # Output shape: (Batch, TotalNumberOfAnchors, 5) (4 for bbox, 1 for depth)
        final_reg_preds = torch.cat(all_reg_preds, dim=1)

        return final_cls_preds, final_reg_preds
