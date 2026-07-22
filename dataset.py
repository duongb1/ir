import os
import glob
import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

class SISMRIDataset(Dataset):
    """
    Dataset loader for S.I.S Brain MRI dataset.
    Loads 2D JPG slice series from S.I.S/images/{STT}/{sequences} and pairs them with text reports.
    """
    def __init__(
        self,
        report_path,
        images_dir,
        text_column="KETLUAN",
        id_column="STT",
        sequences=("DWI",),  # Can be ("DWI",), ("ADC",), or ("DWI", "ADC")
        target_image_size=(480, 480),
        target_depth=240,
        is_train=True,
        need_aug=False
    ):
        super().__init__()
        self.report_path = report_path
        self.images_dir = images_dir
        self.text_column = text_column
        self.id_column = id_column
        self.sequences = [sequences] if isinstance(sequences, str) else list(sequences)
        self.target_image_size = target_image_size
        self.target_depth = target_depth
        self.is_train = is_train
        self.need_aug = need_aug

        # Load Excel / CSV report
        if report_path.endswith('.xlsx') or report_path.endswith('.xls'):
            self.df = pd.read_excel(report_path)
        else:
            self.df = pd.read_csv(report_path)

        # Build list of valid samples (STT, text_content)
        self.samples = []
        for _, row in self.df.iterrows():
            stt = str(int(row[self.id_column])) if pd.notna(row[self.id_column]) else None
            text = str(row[self.text_column]).strip() if pd.notna(row[self.text_column]) else ""
            
            if stt is not None and text and text.lower() != 'nan':
                # Check if case directory exists
                case_dir = os.path.join(self.images_dir, stt)
                if os.path.exists(case_dir):
                    self.samples.append((stt, text, case_dir))

        print(f"[SISMRIDataset] Loaded {len(self.samples)} valid samples from {report_path}")

    def __len__(self):
        return len(self.samples)

    def load_sequence_slices(self, case_dir, seq_name):
        """
        Loads all JPG image slices for a given sequence (e.g. DWI or ADC).
        """
        seq_dir = os.path.join(case_dir, seq_name)
        if not os.path.exists(seq_dir):
            return None

        # Find all jpg / png images in folder
        img_paths = sorted(
            glob.glob(os.path.join(seq_dir, "*.JPG")) +
            glob.glob(os.path.join(seq_dir, "*.jpg")) +
            glob.glob(os.path.join(seq_dir, "*.png"))
        )
        
        if len(img_paths) == 0:
            return None

        slice_tensors = []
        target_h, target_w = self.target_image_size

        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert('L') # Convert to grayscale
                if img.size != (target_w, target_h):
                    img = img.resize((target_w, target_h), Image.BILINEAR)
                
                # Normalize pixel values to [-1, 1]
                arr = (np.array(img, dtype=np.float32) / 255.0) * 2.0 - 1.0
                slice_tensors.append(torch.from_numpy(arr))
            except Exception as e:
                continue

        if len(slice_tensors) == 0:
            return None

        # Stack slices along depth dimension: shape [Depth, Height, Width]
        volume = torch.stack(slice_tensors, dim=0)
        return volume

    def pad_or_crop_depth(self, volume):
        """
        Pads or crops the depth dimension of volume [Depth, Height, Width] to target_depth.
        """
        curr_d, h, w = volume.shape
        target_d = self.target_depth

        if curr_d == target_d:
            return volume
        elif curr_d > target_d:
            # Crop center depth
            start = (curr_d - target_d) // 2
            return volume[start:start + target_d, :, :]
        else:
            # Pad depth evenly on both sides
            pad_before = (target_d - curr_d) // 2
            pad_after = target_d - curr_d - pad_before
            # F.pad expects padding for (last_dim, ..., first_dim) -> (w, h, depth)
            volume = volume.unsqueeze(0).unsqueeze(0) # [1, 1, Depth, Height, Width]
            padded = F.pad(volume, (0, 0, 0, 0, pad_before, pad_after), value=-1.0)
            return padded.squeeze(0).squeeze(0)

    def apply_3d_aug(self, volume_tensor):
        """
        Applies medical-safe 3D data augmentation to volume_tensor [1, Depth, Height, Width]:
        - NO Horizontal/Vertical Flips (preserves left/right medical semantics)
        - Contrast scaling (0.9 to 1.1)
        - Additive Gaussian Noise (sigma = 0.02)
        - Small random translation (up to 5% shift)
        """
        # 1. Random Contrast Scaling
        if torch.rand(1).item() > 0.5:
            scale = torch.empty(1).uniform_(0.9, 1.1).item()
            volume_tensor = torch.clamp(volume_tensor * scale, -1.0, 1.0)

        # 2. Additive Gaussian Noise
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(volume_tensor) * 0.02
            volume_tensor = torch.clamp(volume_tensor + noise, -1.0, 1.0)

        # 3. Small Random Translation (up to 5% shift)
        if torch.rand(1).item() > 0.5:
            _, d, h, w = volume_tensor.shape
            max_shift_h = max(1, int(h * 0.05))
            max_shift_w = max(1, int(w * 0.05))
            shift_h = int(torch.randint(-max_shift_h, max_shift_h + 1, (1,)).item())
            shift_w = int(torch.randint(-max_shift_w, max_shift_w + 1, (1,)).item())
            volume_tensor = torch.roll(volume_tensor, shifts=(shift_h, shift_w), dims=(2, 3))

        return volume_tensor

    def __getitem__(self, idx):
        stt, text_str, case_dir = self.samples[idx]

        # Load MRI volume from configured sequences
        volumes = []
        for seq_name in self.sequences:
            vol = self.load_sequence_slices(case_dir, seq_name)
            if vol is not None:
                vol = self.pad_or_crop_depth(vol)
                volumes.append(vol)

        if len(volumes) == 0:
            # Fallback tensor if images could not be loaded
            volume_tensor = torch.zeros((1, self.target_depth, self.target_image_size[0], self.target_image_size[1]), dtype=torch.float32)
        else:
            # If multiple sequences, average or stack them. Default: average or take primary sequence
            volume_tensor = torch.stack(volumes, dim=0).mean(dim=0) # [Depth, Height, Width]
            volume_tensor = volume_tensor.unsqueeze(0) # [1, Depth, Height, Width]

        # Apply medical-safe 3D data augmentation during training
        if self.is_train:
            volume_tensor = self.apply_3d_aug(volume_tensor)

        # Format text string
        text_str = str(text_str).strip()

        # Modality index (0 for 3D MRI)
        modality_idx = 0

        return volume_tensor, text_str, idx, modality_idx
