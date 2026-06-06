"""
train.py
--------
Fine-tuning pipeline for multilingual fake news detection.

Features:
  - Linear warmup + cosine decay LR schedule
  - Class-weighted loss for imbalanced labels
  - Per-language validation breakdown
  - Best model checkpointing
  - Early stopping

Usage:
    python src/train.py --model xlmroberta --epochs 5 --batch_size 16
"""

import os
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model
from evaluate import compute_metrics, per_language_metrics


# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,   default="xlmroberta",
                        choices=["xlmroberta", "mbert"])
    parser.add_argument("--data_dir",   type=str,   default="data/processed")
    parser.add_argument("--save_dir",   type=str,   default="models/best_model")
    parser.add_argument("--results_dir",type=str,   default="results")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_length", type=int,   default=256)
    parser.add_argument("--warmup_ratio",type=float,default=0.1)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=3)
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_langs = [], [], []

    pbar = tqdm(loader, desc=f"[Epoch {epoch}] Train", leave=False)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        langs          = batch["lang"]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — important for transformer fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_langs.extend(langs)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics, all_labels, all_preds, all_langs


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, split="Val"):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_langs = [], [], []

    for batch in tqdm(loader, desc=f"[Epoch {epoch}] {split}", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * input_ids.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_langs.extend(batch["lang"])

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics, all_labels, all_preds, all_langs


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Multilingual Fake News Detection — Training")
    print(f"  Model: {args.model} | Device: {device}")
    print(f"{'='*55}\n")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Model name map
    tokenizer_map = {
        "xlmroberta": "xlm-roberta-base",
        "mbert":      "bert-base-multilingual-cased",
    }
    tokenizer_name = tokenizer_map[args.model]

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        tokenizer_name=tokenizer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # Model
    model = build_model(args.model, dropout=args.dropout).to(device)

    # Class-weighted loss
    class_weights = train_loader.dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # CSV log
    log_path = Path(args.results_dir) / "training_log.csv"
    log_fields = ["epoch", "train_loss", "train_acc", "train_f1",
                  "val_loss", "val_acc", "val_f1"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_val_f1     = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics, _, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        val_metrics, val_labels, val_preds, val_langs = evaluate(
            model, val_loader, criterion, device, epoch
        )

        elapsed = time.time() - t0

        # Per-language breakdown on val
        lang_metrics = per_language_metrics(val_labels, val_preds, val_langs)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"{elapsed:.1f}s"
        )
        for lang, lm in lang_metrics.items():
            print(f"  [{lang}] Acc: {lm['accuracy']:.4f} F1: {lm['f1']:.4f}")

        # Log
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch":      epoch,
                "train_loss": round(train_metrics["loss"], 4),
                "train_acc":  round(train_metrics["accuracy"], 4),
                "train_f1":   round(train_metrics["f1"], 4),
                "val_loss":   round(val_metrics["loss"], 4),
                "val_acc":    round(val_metrics["accuracy"], 4),
                "val_f1":     round(val_metrics["f1"], 4),
            })

        # Checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            model.save_pretrained(args.save_dir)
            print(f"  ✓ Best model saved (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stop] No improvement for {args.patience} epochs.")
                break

    print(f"\n{'='*55}")
    print(f"  Training complete! Best Val F1: {best_val_f1:.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
