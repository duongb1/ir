import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

from dataset import SISMRIDataset
from model import build_stage1_model
from evaluate import evaluate_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 RadIR model on S.I.S dataset (Kaggle 2x T4 Optimized)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--report_path", type=str, default=None, help="Override report_path")
    parser.add_argument("--images_dir", type=str, default=None, help="Override images_dir")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override lr")
    parser.add_argument("--single_gpu", action="store_true", help="Force single GPU usage on cuda:0")
    args = parser.parse_args()

    if args.single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[Train] Forcing single GPU mode on CUDA_VISIBLE_DEVICES=0")

    # Load configuration
    config_path = args.config if os.path.exists(args.config) else os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply overrides if provided
    if args.report_path: config['report_path'] = args.report_path
    if args.images_dir: config['images_dir'] = args.images_dir
    if args.output_dir: config['output_dir'] = args.output_dir
    if args.epochs: config['epochs'] = args.epochs
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.lr: config['lr'] = args.lr

    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"[Train] Device: {device} | Available GPUs: {num_gpus}")

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # 1. Initialize Full Dataset
    full_dataset = SISMRIDataset(
        report_path=config['report_path'],
        images_dir=config['images_dir'],
        text_column=config['text_column'],
        id_column=config['id_column'],
        sequences=config['sequences'],
        target_image_size=tuple(config['target_image_size']),
        target_depth=config['target_depth'],
        is_train=True
    )

    # Split into Train / Val / Test sets
    total_len = len(full_dataset)
    val_len = int(total_len * config.get('val_ratio', 0.15))
    test_len = int(total_len * config.get('test_ratio', 0.15))
    train_len = total_len - val_len - test_len

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )

    print(f"[Train] Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    num_workers = config.get('num_workers', 2)
    per_gpu_batch = config['batch_size']
    total_batch_per_step = per_gpu_batch * max(1, num_gpus)
    grad_accum_steps = config.get('gradient_accumulation_steps', 4)

    print(f"[Train] Per-GPU Batch Size: {per_gpu_batch} | Effective Step Batch Size: {total_batch_per_step} | Grad Accum Steps: {grad_accum_steps} (Effective Total Batch: {total_batch_per_step * grad_accum_steps})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=total_batch_per_step,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=total_batch_per_step,
        shuffle=False,
        num_workers=num_workers
    )

    # 2. Build Model & Tokenizer
    raw_model, tokenizer = build_stage1_model(
        model_name_or_path=config['text_model_name'],
        dim_text=config['dim_text'],
        dim_image=config['dim_image'],
        dim_latent=config['dim_latent'],
        resnet_depth=config.get('resnet_depth', 18),
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        temporal_patch_size=config['temporal_patch_size'],
        use_triplet_loss=config['use_triplet_loss'],
        use_infoNCE_loss=config['use_infoNCE_loss'],
        device=device
    )

    if config.get('freeze_text_encoder', False):
        print("[Train] Freezing PhoBERT text encoder weights to optimize VRAM")
        raw_model.freeze_text_encoder()

    # Wrap model in DataParallel for 2x T4 GPUs if available
    if num_gpus > 1:
        print(f"[Train] Wrapping model in DataParallel for {num_gpus} GPUs")
        model = nn.DataParallel(raw_model)
    else:
        model = raw_model

    # Differential Learning Rates: lower LR for pre-trained backbones, higher LR for projection heads
    backbone_keywords = ['text_transformer', 'visual_transformer']
    backbone_params = [p for name, p in raw_model.named_parameters() if p.requires_grad and any(k in name for k in backbone_keywords)]
    head_params = [p for name, p in raw_model.named_parameters() if p.requires_grad and not any(k in name for k in backbone_keywords)]

    base_lr = config['lr']
    backbone_lr = config.get('backbone_lr', base_lr * 0.1)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': base_lr}
    ], weight_decay=config.get('weight_decay', 0.01))

    print(f"[Train] Differential LRs: Head LR={base_lr}, Backbone LR={backbone_lr}")

    total_steps = (len(train_loader) // grad_accum_steps) * config['epochs']
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps)
    )

    # Automatic Mixed Precision (AMP FP16) - PyTorch 2.x updated API
    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"[Train] Automatic Mixed Precision (AMP FP16): {'ENABLED' if use_amp else 'DISABLED'}")

    # 3. Training Loop
    print(f"[Train] Starting Stage 1 Training for {config['epochs']} epochs...")
    best_val_r1 = -1.0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (images, text_list, idxs, modality_idxs) in enumerate(train_loader):
            images = images.to(device)
            modal_indices = modality_idxs.to(device)

            # Tokenize batch text with PhoBERT
            text_tokens = tokenizer(
                list(text_list),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            ).to(device)

            # Compute ground-truth text-to-text similarity matrix (T) using PhoBERT embeddings
            with torch.no_grad():
                text_embs = raw_model.text_transformer(text_tokens.input_ids, attention_mask=text_tokens.attention_mask)[0][:, 0, :]
                text_embs = torch.nn.functional.normalize(text_embs.float(), dim=-1)
                gt_sim_matrix = torch.matmul(text_embs, text_embs.T)

            # Forward pass with PyTorch 2.x torch.amp.autocast
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = model(
                    text_tokens,
                    image=images,
                    device=device,
                    gt_similarity_matrix=gt_sim_matrix,
                    is_condition=False,
                    modal_indexs=modal_indices,
                    modal_embedding=True
                )

            if isinstance(loss, tuple):
                loss = loss[0]
            if loss.dim() > 0:
                loss = loss.mean()

            # Normalize loss for gradient accumulation
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer.zero_grad()
                if scale_before <= scale_after:
                    scheduler.step()

            total_loss += loss.item() * grad_accum_steps

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{config['epochs']:02d} | Train Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Evaluation
        if epoch % config.get('eval_every_epochs', 1) == 0 or epoch == config['epochs']:
            t2i_recalls, i2t_recalls = evaluate_model(raw_model, val_loader, tokenizer, device, use_amp=use_amp)
            val_r1 = t2i_recalls.get("R@1", 0.0)

            print(f"   [Val Eval] T2I R@1: {t2i_recalls['R@1']*100:.2f}% | R@5: {t2i_recalls['R@5']*100:.2f}% | R@10: {t2i_recalls['R@10']*100:.2f}%")
            print(f"   [Val Eval] I2T R@1: {i2t_recalls['R@1']*100:.2f}% | R@5: {i2t_recalls['R@5']*100:.2f}% | R@10: {i2t_recalls['R@10']*100:.2f}%")

            # Save best checkpoint
            if val_r1 > best_val_r1:
                best_val_r1 = val_r1
                best_ckpt_path = os.path.join(config['output_dir'], "best_stage1_model.pt")
                torch.save(raw_model.state_dict(), best_ckpt_path)
                print(f"   [Checkpoint] Best model saved to {best_ckpt_path} (T2I R@1={val_r1*100:.2f}%)")

        # Save periodic checkpoint
        if epoch % config.get('save_every_epochs', 5) == 0:
            ckpt_path = os.path.join(config['output_dir'], f"stage1_epoch_{epoch}.pt")
            torch.save(raw_model.state_dict(), ckpt_path)

    print("\n[Train] Training complete!")

if __name__ == '__main__':
    main()
