import torch
import torch.nn as nn
import torchvision.models.video as models

class ResNet3DEncoder(nn.Module):
    def __init__(self, dim=512, depth=50, pretrained=True):
        super(ResNet3DEncoder, self).__init__()
        self.depth = depth
        if depth == 50:
            from torchvision.models.video.resnet import VideoResNet, Bottleneck, Conv3DSimple, BasicStem
            self.base_model = VideoResNet(
                block=Bottleneck,
                conv_makers=[Conv3DSimple]*4,
                layers=[3, 4, 6, 3],
                stem=BasicStem
            )
            in_features = 2048
        else:
            weights = models.R3D_18_Weights.DEFAULT if pretrained else None
            self.base_model = models.r3d_18(weights=weights)
            in_features = 512
        
        # Remove the classification head (fc layer)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Projection head to map to the desired output dimension
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, dim)
        )

    def forward(self, x, return_encoded_tokens=False, modal_embedding=False, modal_indexs=None, is_condition=False):
        """
        x: [B, C, D, H, W] tensor (e.g. [B, 1, 24, 256, 256])
        """
        # Ensure x is 5D: [B, C, D, H, W]
        # Our dataset returns [1, 240, 480, 480], need to reshape or ensure C=1
        if x.ndim == 4: # [B, D, H, W]
            x = x.unsqueeze(1) # [B, 1, D, H, W]
            
        # ResNet3D expects 3 channels, so we can duplicate the 1 channel to 3
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)

        # Forward through ResNet3D
        out = self.base_model(x) # [B, 512, 1, 1, 1]
        
        # Project to target dimension
        latent = self.proj(out) # [B, dim]
        
        # If the model requires encoded tokens (e.g., for cross attention in condition mode)
        # We can simulate tokens by just returning the latent expanded, or the raw features
        if return_encoded_tokens:
            # Fake the tokens shape: [B, T, H, W, dim] -> [B, 1, 1, 1, dim]
            return latent.view(latent.shape[0], 1, 1, 1, latent.shape[-1])
            
        return latent

if __name__ == '__main__':
    # Test
    model = ResNet3DEncoder(dim=512, pretrained=False)
    x = torch.randn(2, 1, 24, 256, 256)
    out = model(x)
    print("Output shape:", out.shape)
