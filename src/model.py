"""
model.py
--------
Model definitions:
  - BaselineModel:     TF-IDF + Logistic Regression (classical ML baseline)
  - TransformerModel:  XLM-RoBERTa / mBERT with custom 3-class head
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import os


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

LABEL2ID = {"real": 0, "fake": 1, "satire": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 3


# ─────────────────────────────────────────────
# 1. Baseline: TF-IDF + Logistic Regression
# ─────────────────────────────────────────────

class BaselineModel:
    """
    Classical ML baseline: TF-IDF features + Logistic Regression.
    Fast to train, good for English but weaker on Uzbek/Russian.
    Used to establish a lower-bound benchmark.
    """

    def __init__(self, max_features: int = 50000, ngram_range=(1, 2)):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,          # Log-scale TF
                analyzer="word",
                strip_accents="unicode",
                min_df=2,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",    # Handle class imbalance
                multi_class="multinomial",
                solver="lbfgs",
            )),
        ])

    def fit(self, texts, labels):
        print("[Baseline] Training TF-IDF + Logistic Regression...")
        self.pipeline.fit(texts, labels)
        print("[Baseline] Training complete.")
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

    def save(self, path: str = "models/baseline.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"[Baseline] Saved to {path}")

    def load(self, path: str = "models/baseline.pkl"):
        self.pipeline = joblib.load(path)
        return self


# ─────────────────────────────────────────────
# 2. Transformer model (XLM-RoBERTa / mBERT)
# ─────────────────────────────────────────────

class TransformerModel(nn.Module):
    """
    Fine-tuned multilingual transformer for 3-class fake news detection.

    Architecture:
        XLM-RoBERTa-base (or mBERT)
            ↓
        [CLS] token (768-dim)
            ↓
        Dropout(0.3)
            ↓
        FC(768 → 256) + GELU + Dropout(0.2)
            ↓
        FC(256 → 3) + Softmax

    Args:
        model_name:  HuggingFace model identifier
        num_classes: Number of output classes (3)
        dropout:     Dropout rate on [CLS] representation
        freeze_base: If True, freeze transformer body (train head only)
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.model_name = model_name

        # Load pre-trained transformer
        self.config    = AutoConfig.from_pretrained(model_name)
        self.backbone  = AutoModel.from_pretrained(model_name)
        hidden_size    = self.config.hidden_size    # 768 for base models

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),             # Slightly lower dropout in middle
            nn.Linear(256, num_classes),
        )

        if freeze_base:
            self.freeze_backbone()

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] {model_name} | Total: {total:,} | Trainable: {trainable:,}")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen.")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] Backbone unfrozen.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]   # (B, hidden_size)
        logits     = self.classifier(cls_output)           # (B, num_classes)
        return logits

    def save_pretrained(self, save_dir: str = "models/best_model"):
        os.makedirs(save_dir, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, "classifier_head.pt"))
        print(f"[Model] Saved to {save_dir}/")

    def load_pretrained(self, save_dir: str = "models/best_model", device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = AutoModel.from_pretrained(save_dir).to(device)
        head_path = os.path.join(save_dir, "classifier_head.pt")
        self.classifier.load_state_dict(torch.load(head_path, map_location=device))
        return self


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_model(model_type: str = "xlmroberta", **kwargs) -> nn.Module:
    model_map = {
        "xlmroberta": "xlm-roberta-base",
        "mbert":      "bert-base-multilingual-cased",
    }
    if model_type == "baseline":
        return BaselineModel(**kwargs)

    assert model_type in model_map, f"model_type must be one of {list(model_map.keys()) + ['baseline']}"
    return TransformerModel(model_name=model_map[model_type], **kwargs)


if __name__ == "__main__":
    # Sanity check
    model = build_model("xlmroberta")
    dummy_ids  = torch.randint(0, 250002, (2, 64))
    dummy_mask = torch.ones(2, 64, dtype=torch.long)
    out = model(dummy_ids, dummy_mask)
    print(f"Output shape: {out.shape}")   # (2, 3)
