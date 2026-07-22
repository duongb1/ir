import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VisionMIL(nn.Module):
    """
    2D Attention-based Multiple Instance Learning (ABMIL) Vision Encoder.
    Extracts slice features using a 2D ResNet backbone (ImageNet pre-trained),
    applies slice attention pooling with masking, and projects to latent dimension.
    """
    def __init__(self, embed_dim=512, depth=18, pretrained=True):
        super(VisionMIL, self).__init__()
        self.depth = depth
        if depth == 50:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.feat_dim = 2048
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.feat_dim = 512

        # 1. 2D CNN Feature Extractor (Remove FC layer)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # 2. Attention Pooling Module
        self.attention = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # 3. Projection Head
        self.projector = nn.Linear(self.feat_dim, embed_dim)

    def forward(self, x, mask=None, return_encoded_tokens=False, modal_embedding=False, modal_indexs=None, is_condition=False):
        """
        x: [B, N, C, H, W] tensor (N = number of 2D slices, C = 3 channels: ADC, DWI, DWI-ADC)
        mask: [B, N] boolean tensor (True for valid slices, False for padded slices)
        """
        # Handle [B, C, N, H, W] shape if passed
        if x.ndim == 5 and x.shape[1] in [1, 3]:
            x = x.permute(0, 2, 1, 3, 4)

        B, N, C, H, W = x.shape

        # Reshape to [B*N, C, H, W] for 2D CNN forward pass
        x_flat = x.reshape(B * N, C, H, W)
        
        # Extract features per slice
        feats = self.feature_extractor(x_flat) # [B*N, feat_dim, 1, 1]
        feats = feats.view(B, N, self.feat_dim) # [B, N, feat_dim]

        # Compute raw attention scores per slice
        attn_scores = self.attention(feats).squeeze(-1) # [B, N]

        # Apply Attention Masking (-1e9 for padded slices)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e9)

        # Softmax over valid slices
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1) # [B, N, 1]

        # Aggregate slice features weighted by Attention
        patient_feature = torch.sum(feats * attn_weights, dim=1) # [B, feat_dim]

        # Project to target latent dimension (512)
        latent = self.projector(patient_feature) # [B, embed_dim]

        if return_encoded_tokens:
            return latent.view(B, 1, 1, 1, latent.shape[-1])

        return latent

if __name__ == '__main__':
    model = VisionMIL(embed_dim=512, depth=18, pretrained=False)
    x = torch.randn(2, 10, 3, 256, 256)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, 5:] = False # Padded slices 5..9
    out = model(x, mask=mask)
    print("VisionMIL output shape:", out.shape)
