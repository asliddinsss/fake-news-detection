"""
dataset.py
----------
Dataset class and preprocessing pipeline for multilingual fake news detection.

Supports:
  - Loading from CSV (unified format across all source datasets)
  - Language detection and filtering
  - HuggingFace tokenization (XLM-RoBERTa / mBERT)
  - Class-weighted sampling for imbalanced data
  - Back-translation augmentation placeholder

Label mapping:
  0 → Real
  1 → Fake
  2 → Satire
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from langdetect import detect, LangDetectException


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

LABEL2ID = {"real": 0, "fake": 1, "satire": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LANG_MAP = {"en": "English", "ru": "Russian", "uz": "Uzbek"}

SUPPORTED_LANGS = {"en", "ru", "uz"}


# ─────────────────────────────────────────────
# Text preprocessing
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text cleaning — remove excess whitespace and control chars."""
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())          # Normalize whitespace
    text = text.strip()
    return text


def detect_language(text: str) -> Optional[str]:
    """Detect language of a text snippet. Returns ISO code or None."""
    try:
        lang = detect(text[:500])          # Only use first 500 chars for speed
        return lang if lang in SUPPORTED_LANGS else None
    except LangDetectException:
        return None


# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────

class FakeNewsDataset(Dataset):
    """
    PyTorch Dataset for multilingual fake news classification.

    Expected CSV columns:
        text  : article text or headline
        label : 'real' | 'fake' | 'satire'
        lang  : 'en' | 'ru' | 'uz'  (optional — auto-detected if missing)

    Args:
        csv_path:       Path to CSV file
        tokenizer_name: HuggingFace model name for tokenizer
        max_length:     Maximum token length
        lang_filter:    If set, only load samples of this language
        augment:        If True, apply simple augmentation (token dropout)
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "xlm-roberta-base",
        max_length: int = 256,
        lang_filter: Optional[str] = None,
        augment: bool = False,
    ):
        self.max_length = max_length
        self.augment = augment

        # Load tokenizer
        print(f"[Dataset] Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load and preprocess data
        df = pd.read_csv(csv_path)
        df = self._preprocess(df, lang_filter)
        self.texts  = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.langs  = df["lang"].tolist()

        print(f"[Dataset] Loaded {len(self.texts)} samples from {csv_path}")
        print(f"  Label dist: { {ID2LABEL[i]: self.labels.count(i) for i in range(3)} }")
        if lang_filter:
            print(f"  Language filter: {lang_filter}")

    def _preprocess(self, df: pd.DataFrame, lang_filter: Optional[str]) -> pd.DataFrame:
        # Clean text
        df["text"] = df["text"].apply(clean_text)
        df = df[df["text"].str.len() > 20]          # Drop near-empty texts

        # Normalize labels
        df["label"] = df["label"].str.lower().str.strip()
        df = df[df["label"].isin(LABEL2ID)]
        df["label_id"] = df["label"].map(LABEL2ID)

        # Language column
        if "lang" not in df.columns:
            print("[Dataset] No lang column found — auto-detecting...")
            df["lang"] = df["text"].apply(detect_language)

        df = df[df["lang"].isin(SUPPORTED_LANGS)]

        if lang_filter:
            df = df[df["lang"] == lang_filter]

        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text  = self.texts[idx]
        label = self.labels[idx]

        # Simple token dropout augmentation
        if self.augment and np.random.rand() < 0.15:
            words = text.split()
            keep  = np.random.rand(len(words)) > 0.1
            text  = " ".join([w for w, k in zip(words, keep) if k])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
            "lang":           self.langs[idx],
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted loss."""
        counts = np.bincount(self.labels, minlength=3).astype(float)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * 3   # Normalize to sum to num_classes
        return torch.tensor(weights, dtype=torch.float)

    def get_sampler(self) -> WeightedRandomSampler:
        """Weighted sampler to handle class imbalance during training."""
        class_weights = self.get_class_weights().numpy()
        sample_weights = [class_weights[l] for l in self.labels]
        return WeightedRandomSampler(sample_weights, num_samples=len(self.labels), replacement=True)


# ─────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate — handles the 'lang' string field."""
    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label":          torch.stack([b["label"] for b in batch]),
        "lang":           [b["lang"] for b in batch],
    }


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def get_dataloaders(
    data_dir: str = "data/processed",
    tokenizer_name: str = "xlm-roberta-base",
    max_length: int = 256,
    batch_size: int = 16,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader)."""

    train_ds = FakeNewsDataset(
        os.path.join(data_dir, "train.csv"),
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        augment=True,
    )
    val_ds = FakeNewsDataset(
        os.path.join(data_dir, "val.csv"),
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        augment=False,
    )
    test_ds = FakeNewsDataset(
        os.path.join(data_dir, "test.csv"),
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_ds.get_sampler(),          # Weighted sampling for class balance
        num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
