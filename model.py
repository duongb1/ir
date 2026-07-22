import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from resnet3d import ResNet3DEncoder
from radir import RADIR

def build_stage1_model(
    model_name_or_path="vinai/phobert-base",
    dim_text=768,
    dim_image=512,
    dim_latent=512,
    resnet_depth=18,
    image_size=384,
    patch_size=16,
    temporal_patch_size=4,
    spatial_depth=8,
    temporal_depth=6,
    cls_depth=4,
    dim_head=32,
    heads=8,
    use_triplet_loss=1.0,
    use_infoNCE_loss=1.0,
    device=None
):
    """
    Builds and initializes the Stage 1 RadIR model using ResNet3D (Vision) and PhoBERT (Text).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load PhoBERT Tokenizer and Text Encoder
    print(f"[Model] Loading PhoBERT text encoder: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    text_encoder = AutoModel.from_pretrained(model_name_or_path)

    # 2. Build 3D Vision Encoder (ResNet3D)
    print(f"[Model] Initializing ResNet3D-{resnet_depth} 3D Vision Encoder")
    image_encoder = ResNet3DEncoder(
        dim=dim_image,
        depth=resnet_depth,
        pretrained=True
    )

    # 3. Instantiate RADIR Stage 1 Model
    print(f"[Model] Wrapping into RADIR Stage 1 framework (dim_latent={dim_latent})")
    radir_model = RADIR(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        use_triplet_loss=use_triplet_loss,
        use_infoNCE_loss=use_infoNCE_loss,
        use_uncon_triplet_loss=use_triplet_loss,
        use_uncon_infoNCE_loss=use_infoNCE_loss,
        dim_text=dim_text,      # PhoBERT default is 768
        dim_image=dim_image,    # CTViT default is 512
        dim_latent=dim_latent,  # Joint latent space 512
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False
    )

    radir_model.to(device)
    return radir_model, tokenizer
