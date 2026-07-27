import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import SISMRIDataset, sis_collate_fn
from model import build_stage1_model

def compute_retrieval_metrics(image_embeds, text_embeds, all_texts, k_values=(1, 5, 10)):
    """
    Computes Recall@K metrics for Text-to-Image (T2I) and Image-to-Text (I2T) retrieval.
    Includes Multi-Ground-Truth (Multi-GT) support to handle duplicate text reports.
    """
    # Calculate similarity matrix: [N, N]
    sim_matrix = torch.matmul(image_embeds, text_embeds.T)
    N = sim_matrix.shape[0]

    # --- TẠO MA TRẬN MULTI-GROUND-TRUTH ---
    # Sử dụng numpy broadcasting để so sánh chuỗi cực nhanh
    texts_np = np.array(all_texts)
    # gt_matrix[i, j] = True nếu text của bệnh nhân i giống hệt text bệnh nhân j
    gt_matrix_np = (texts_np[:, None] == texts_np[None, :])
    gt_matrix = torch.tensor(gt_matrix_np, device=sim_matrix.device)

    # 1. Text-to-Image Retrieval (Query: Text, Target: Image)
    t2i_sim = sim_matrix.T  # [N_text, N_image]
    t2i_ranks = torch.argsort(t2i_sim, dim=1, descending=True)
    t2i_recalls = {}
    for k in k_values:
        # Lấy top k chỉ mục ảnh được truy xuất
        topk_indices = t2i_ranks[:, :k] # [N, K]
        # Đối chiếu với ma trận GT xem có bất kỳ ảnh nào khớp nội dung không
        hits = torch.gather(gt_matrix, 1, topk_indices).any(dim=1)
        t2i_recalls[f"R@{k}"] = hits.float().mean().item()

    # 2. Image-to-Text Retrieval (Query: Image, Target: Text)
    i2t_sim = sim_matrix    # [N_image, N_text]
    i2t_ranks = torch.argsort(i2t_sim, dim=1, descending=True)
    i2t_recalls = {}
    for k in k_values:
        topk_indices = i2t_ranks[:, :k] # [N, K]
        # Ma trận GT đối xứng nên cách tính tương tự
        hits = torch.gather(gt_matrix, 1, topk_indices).any(dim=1)
        i2t_recalls[f"R@{k}"] = hits.float().mean().item()

    return t2i_recalls, i2t_recalls


@torch.no_grad()
def evaluate_model(model, dataloader, tokenizer, device, use_amp=True):
    """
    Evaluates model on dataloader by extracting embeddings and calculating Recall@K metrics.
    """
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    all_texts = [] # Khởi tạo danh sách lưu trữ text gốc

    for step, batch in enumerate(dataloader):
        if len(batch) == 5:
            images, text_list, idxs, modality_idxs, mask = batch
            mask = mask.to(device, non_blocking=True)
        else:
            images, text_list, idxs, modality_idxs = batch
            mask = None

        images = images.to(device)
        modal_indices = modality_idxs.to(device)
        
        # Lưu lại text gốc của batch hiện tại để xây dựng GT Matrix
        all_texts.extend(text_list)

        # Tokenize text using PhoBERT tokenizer
        text_tokens = tokenizer(
            list(text_list),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(device)

        # Forward pass through RADIR with AMP context
        with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
            text_emb, image_emb, _, _ = model(
                text_tokens,
                image=images,
                mask=mask,
                device=device,
                is_condition=False,
                return_latents=True,
                modal_indexs=modal_indices,
                modal_embedding=True
            )

        # Normalize L2
        image_emb = torch.nn.functional.normalize(image_emb.float(), dim=-1)
        text_emb = torch.nn.functional.normalize(text_emb.float(), dim=-1)

        all_image_embeds.append(image_emb.cpu())
        all_text_embeds.append(text_emb.cpu())

    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)

    # Truyền thêm all_texts vào hàm tính toán độ đo
    t2i_recalls, i2t_recalls = compute_retrieval_metrics(image_embeds, text_embeds, all_texts, k_values=(1, 5, 10))

    return t2i_recalls, i2t_recalls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to model checkpoint (.pt)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build dataset & loader
    dataset = SISMRIDataset(
        report_path=config['report_path'],
        images_dir=config['images_dir'],
        text_column=config['text_column'],
        id_column=config['id_column'],
        sequences=config['sequences'],
        target_image_size=tuple(config['target_image_size']),
        target_depth=config['target_depth'],
        is_train=False
    )
    
    # Sử dụng custom collate_fn để tương thích với mask của kiến trúc 2D MIL
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config.get('num_workers', 4), collate_fn=sis_collate_fn)

    # Build model
    model, tokenizer = build_stage1_model(
        model_name_or_path=config['text_model_name'],
        dim_text=config['dim_text'],
        dim_image=config['dim_image'],
        dim_latent=config['dim_latent'],
        resnet_depth=config.get('resnet_depth', 50), # Điều chỉnh theo 2D Backbone (18/50)
        device=device,
        vision_type=config.get('vision_type', 'resnet3d') # Dự phòng cho bản cập nhật mới
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[Evaluate] Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    t2i_recalls, i2t_recalls = evaluate_model(model, dataloader, tokenizer, device, use_amp=config.get('use_amp', True))

    print("\n" + "="*50)
    print("EVALUATION RESULTS (S.I.S Stage 1 Retrieval - Multi-GT)")
    print("="*50)
    print("Text-to-Image Retrieval (T2I):")
    for k, v in t2i_recalls.items():
        print(f"  {k}: {v * 100:.2f}%")
    print("\nImage-to-Text Retrieval (I2T):")
    for k, v in i2t_recalls.items():
        print(f"  {k}: {v * 100:.2f}%")
    print("="*50)
