import os
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
    parser = argparse.ArgumentParser(description="Train Stage 1 RadIR model on S.I.S dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--report_path", type=str, default=None, help="Override report_path")
    parser.add_argument("--images_dir", type=str, default=None, help="Override images_dir")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override lr")
    args = parser.parse_args()

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
    print(f"[Train] Using device: {device}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # 2. Build Model & Tokenizer
    model, tokenizer = build_stage1_model(
        model_name_or_path=config['text_model_name'],
        dim_text=config['dim_text'],
        dim_image=config['dim_image'],
        dim_latent=config['dim_latent'],
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        temporal_patch_size=config['temporal_patch_size'],
        use_triplet_loss=config['use_triplet_loss'],
        use_infoNCE_loss=config['use_infoNCE_loss'],
        device=device
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01)
    )

    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 3. Training Loop
    print(f"[Train] Starting Stage 1 Training for {config['epochs']} epochs...")
    best_val_r1 = -1.0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0.0

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

            optimizer.zero_grad()

            # Forward pass: RADIR computes Contrastive + Triplet Loss in forward pass
            loss = model(
                text_tokens,
                image=images,
                device=device,
                is_condition=False,
                modal_indexs=modal_indices,
                modal_embedding=True
            )

            if isinstance(loss, tuple):
                loss = loss[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{config['epochs']:02d} | Train Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Evaluation
        if epoch % config.get('eval_every_epochs', 1) == 0 or epoch == config['epochs']:
            t2i_recalls, i2t_recalls = evaluate_model(model, val_loader, tokenizer, device)
            val_r1 = t2i_recalls.get("R@1", 0.0)

            print(f"   [Val Eval] T2I R@1: {t2i_recalls['R@1']*100:.2f}% | R@5: {t2i_recalls['R@5']*100:.2f}% | R@10: {t2i_recalls['R@10']*100:.2f}%")
            print(f"   [Val Eval] I2T R@1: {i2t_recalls['R@1']*100:.2f}% | R@5: {i2t_recalls['R@5']*100:.2f}% | R@10: {i2t_recalls['R@10']*100:.2f}%")

            # Save best checkpoint
            if val_r1 > best_val_r1:
                best_val_r1 = val_r1
                best_ckpt_path = os.path.join(config['output_dir'], "best_stage1_model.pt")
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"   [Checkpoint] Best model saved to {best_ckpt_path} (T2I R@1={val_r1*100:.2f}%)")

        # Save periodic checkpoint
        if epoch % config.get('save_every_epochs', 5) == 0:
            ckpt_path = os.path.join(config['output_dir'], f"stage1_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

    print("\n[Train] Training complete!")

if __name__ == '__main__':
    main()
