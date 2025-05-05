# # full_model.py
#
# import torch
# import torch.nn as nn
# import timm
# from torch_transformer import AffineTransformer, ProjectiveTransformer  # or ProjectiveTransformer
#
# class FullModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
#         self.fc = nn.Linear(self.backbone.num_features, 8)  # 6 for affine transformer, 8 for projective
#         # self.affine_transformer = AffineTransformer((400, 400))
#         self.projective_transformer = ProjectiveTransformer((400, 400))
#
#     def forward(self, x, ref_img):
#         features = self.backbone(x)
#         theta = self.fc(features)  # [batch, 6]
#         # out = self.affine_transformer(ref_img.expand(x.size(0), -1, -1, -1), theta)
#         out = self.projective_transformer(ref_img.expand(x.size(0), -1, -1, -1), theta)
#         return out

# full_model.py

import torch
import torch.nn as nn
import timm
from torch_transformer import ProjectiveTransformer

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.fc = nn.Linear(self.backbone.num_features, 8)
        self.projective_transformer = ProjectiveTransformer((400, 400))

        # Initialize like TensorFlow (identity transform)
        nn.init.zeros_(self.fc.weight)
        self.fc.bias.data = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float)

    def forward(self, x, ref_img):
        features = self.backbone(x)
        theta = self.fc(features)
        ref_img = ref_img.expand(x.size(0), -1, -1, -1)
        out = self.projective_transformer(ref_img, theta)
        return out
