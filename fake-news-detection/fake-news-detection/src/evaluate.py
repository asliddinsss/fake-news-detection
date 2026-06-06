"""
evaluate.py
-----------
Evaluation utilities:
  - compute_metrics: accuracy, macro F1, per-class F1
  - per_language_metrics: breakdown by language
  - plot_confusion_matrix: 3-class heatmap
  - plot_training_curves
  - plot_per_language_breakdown
"""

import os
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

LABEL_NAMES = ["Real", "Fake", "Satire"]
LANG_NAMES  = {"en": "English", "ru": "Russian", "uz": "Uzbek"}
LANG_COLORS = {"en": "#4fc3f7", "ru": "#ef5350", "uz": "#66bb6a"}


# ─────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_language_metrics(
    y_true: List[int],
    y_pred: List[int],
    langs:  List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by language."""
    lang_results = defaultdict(lambda: {"true": [], "pred": []})
    for t, p, l in zip(y_true, y_pred, langs):
        lang_results[l]["true"].append(t)
        lang_results[l]["pred"].append(p)

    return {
        lang: compute_metrics(v["true"], v["pred"])
        for lang, v in lang_results.items()
    }


# ─────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true, y_pred,
    save_path: str = "results/confusion_matrix.png"
):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#0f1117")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".1%"],
        ["Counts", "Normalized (row %)"],
    ):
        ax.set_facecolor("#1a1d27")
        sns.heatmap(
            data, annot=True, fmt=fmt, ax=ax,
            cmap="YlOrRd",
            linewidths=1.5, linecolor="#0f1117",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            annot_kws={"size": 14, "weight": "bold", "color": "white"},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", color="#aaaaaa", fontsize=11)
        ax.set_ylabel("Actual",    color="#aaaaaa", fontsize=11)
        ax.tick_params(colors="#cccccc", labelsize=10)
        ax.yaxis.set_tick_params(rotation=0)
        ax.collections[0].colorbar.ax.tick_params(colors="#aaaaaa")

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fig.suptitle(
        f"Confusion Matrix — Test Set  |  Accuracy: {acc:.1%}  |  Macro F1: {f1:.3f}",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────

def plot_training_curves(
    log_csv:   str = "results/training_log.csv",
    save_path: str = "results/training_curves.png",
):
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")

    for ax, metric, title in zip(
        axes,
        ["loss", "acc", "f1"],
        ["Loss", "Accuracy", "Macro F1"],
    ):
        ax.plot(df["epoch"], df[f"train_{metric}"],
                color="#4fc3f7", linewidth=2.5, label="Train",
                marker="o", markersize=4, markevery=1)
        ax.plot(df["epoch"], df[f"val_{metric}"],
                color="#ef5350", linewidth=2.5, label="Val",
                linestyle="--", marker="s", markersize=4, markevery=1)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", color="#aaaaaa")
        ax.set_ylabel(title,   color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        ax.grid(alpha=0.15, color="white")
        ax.legend(fontsize=9, facecolor="#1a1d27", labelcolor="white", edgecolor="#333344")

    fig.suptitle("XLM-RoBERTa Fine-tuning Curves", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")


# ─────────────────────────────────────────────
# Per-language breakdown chart
# ─────────────────────────────────────────────

def plot_per_language_breakdown(
    lang_metrics: Dict[str, Dict[str, float]],
    save_path: str = "results/per_language_breakdown.png",
):
    langs   = [LANG_NAMES.get(l, l) for l in lang_metrics]
    accs    = [lang_metrics[l]["accuracy"] for l in lang_metrics]
    f1s     = [lang_metrics[l]["f1"]       for l in lang_metrics]
    colors  = [LANG_COLORS.get(l, "#aaaaaa") for l in lang_metrics]

    x = np.arange(len(langs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=colors, alpha=0.9, edgecolor="#0f1117")
    bars2 = ax.bar(x + w/2, f1s,  w, label="Macro F1", color=colors, alpha=0.55, edgecolor="#0f1117")

    for bar, val in zip(list(bars1) + list(bars2), accs + f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(langs, color="white", fontsize=12)
    ax.set_ylim(0.70, 1.00)
    ax.set_ylabel("Score", color="#aaaaaa")
    ax.set_title("Per-Language Performance — XLM-RoBERTa",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#aaaaaa")
    ax.legend(fontsize=10, facecolor="#1a1d27", labelcolor="white", edgecolor="#333344")
    ax.grid(axis="y", alpha=0.15, color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[Plot] Per-language breakdown saved → {save_path}")


# ─────────────────────────────────────────────
# Full test evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, test_loader, device, results_dir: str = "results"):
    import torch
    model.eval()
    all_preds, all_labels, all_langs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_langs.extend(batch["lang"])

    metrics      = compute_metrics(all_labels, all_preds)
    lang_metrics = per_language_metrics(all_labels, all_preds, all_langs)

    print("\n" + "="*55)
    print("  TEST SET RESULTS")
    print("="*55)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print("="*55)
    print("\nPer-language breakdown:")
    for lang, lm in lang_metrics.items():
        print(f"  [{LANG_NAMES.get(lang, lang)}] Acc: {lm['accuracy']:.4f} | F1: {lm['f1']:.4f}")
    print("\nDetailed report:")
    print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))

    plot_confusion_matrix(all_labels, all_preds,
                          save_path=os.path.join(results_dir, "confusion_matrix.png"))
    plot_per_language_breakdown(lang_metrics,
                                save_path=os.path.join(results_dir, "per_language_breakdown.png"))

    return metrics, lang_metrics
