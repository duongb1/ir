import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class HiSBreastDataset(Dataset):
    """
    Dataset loader for HiSBreast ultrasound dataset.
    Loads 2D PNG images from HiSBreast/images and pairs them with text annotations.
    Formats the output to be compatible with 2D ABMIL architecture (sis_collate_fn).
    """
    def __init__(
        self,
        report_path,
        images_dir,
        text_column="text_description_vn",
        id_column="file_name",
        target_image_size=(256, 256),
        is_train=True,
        need_aug=False
    ):
        super().__init__()
        self.report_path = report_path
        self.images_dir = images_dir
        self.text_column = text_column
        self.id_column = id_column
        self.target_image_size = target_image_size
        self.is_train = is_train
        self.need_aug = need_aug

        # Load CSV report
        self.df = pd.read_csv(report_path)

        # Build list of valid samples (file_name, text_content)
        self.samples = []
        for _, row in self.df.iterrows():
            file_name = str(row[self.id_column]).strip()
            text = str(row[self.text_column]).strip() if pd.notna(row[self.text_column]) else ""
            
            if file_name and text and text.lower() != 'nan':
                img_path = os.path.join(self.images_dir, file_name)
                self.samples.append((file_name, text, img_path))
        
        print(f"[HiSBreastDataset] Loaded {len(self.samples)} valid samples from {report_path}")

        # Data Augmentation & Transforms
        # Chú ý: Tuyệt đối KHÔNG DÙNG RandomHorizontalFlip vì nó làm đảo lộn vị trí trái/phải của tổn thương
        if self.need_aug and self.is_train:
            self.transform = T.Compose([
                T.Resize(target_image_size),
                T.RandomRotation(10), # Quay nhẹ không ảnh hưởng vị trí
                T.ColorJitter(brightness=0.2, contrast=0.2), # Tăng cường ánh sáng/độ tương phản
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Chuẩn hóa [-1, 1]
            ])
        else:
            self.transform = T.Compose([
                T.Resize(target_image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, text_str, img_path = self.samples[idx]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"KHÔNG TÌM THẤY ẢNH: {img_path}\nVui lòng kiểm tra lại cấu trúc thư mục (có thể bị lồng images/images/) hoặc đuôi file (.png hay .jpg).")

        # 1. Mở ảnh và chuyển sang hệ màu RGB (để có 3 kênh tương thích ResNet)
        img = Image.open(img_path).convert('RGB')
        
        # 2. Áp dụng transforms (Resize -> ToTensor [0,1] -> Normalize [-1,1])
        # Đầu ra img_tensor có shape: [3, H, W]
        img_tensor = self.transform(img)
        
        # 3. Tương thích với `sis_collate_fn` của kiến trúc 2D MIL
        # Kiến trúc mong đợi [Depth, C, H, W]. Với ảnh 2D, Depth = 1.
        img_tensor = img_tensor.unsqueeze(0) # Trở thành [1, 3, H, W]

        # Chuẩn hóa chuỗi văn bản
        text_str = str(text_str).strip()

        # Chỉ mục định danh (ở đây dùng số ngẫu nhiên hoặc chính idx vì STT là string)
        # S.I.S Dataset trả về số nguyên idx. Ta dùng idx của mảng.
        numeric_idx = idx

        # Chỉ số Modality (0: placeholder)
        modality_idx = 0

        # Trả về bộ tuple ĐÚNG CHUẨN để `sis_collate_fn` có thể nhận diện và tạo mask
        return img_tensor, text_str, numeric_idx, modality_idx
