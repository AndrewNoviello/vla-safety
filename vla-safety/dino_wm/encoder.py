import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.emb_dim = self.base_model.num_features
        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        features = self.base_model.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]   # (B, N, D)
        class_token = features["x_norm_clstoken"]      # (B, D)
        return {"patch_tokens": patch_tokens, "class_token": class_token}
