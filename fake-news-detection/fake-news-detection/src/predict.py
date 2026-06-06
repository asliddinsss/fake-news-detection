"""
predict.py
----------
Run inference on raw text — single input or batch from CSV.

Usage:
    # Single text
    python src/predict.py --text "Vazirlik yangi qonun loyihasini tasdiqladi" --lang uz

    # Batch from CSV
    python src/predict.py --csv data/samples/sample_articles.csv --output results/predictions.csv
"""

import argparse
import os
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer

from model import TransformerModel
from evaluate import LABEL_NAMES, LANG_NAMES

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

LABEL_EMOJI = {0: "✅", 1: "❌", 2: "🎭"}
MAX_LENGTH  = 256


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────

def load_model(model_dir: str = "models/best_model", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Predict] Loading model from {model_dir}...")
    model     = TransformerModel(model_name=model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    head_path = os.path.join(model_dir, "classifier_head.pt")
    model.classifier.load_state_dict(torch.load(head_path, map_location=device))
    model.eval()

    print(f"[Predict] Model ready on {device}")
    return model, tokenizer, device


# ─────────────────────────────────────────────
# Single prediction
# ─────────────────────────────────────────────

def predict_text(text: str, model, tokenizer, device) -> dict:
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(probs.argmax())
    return {
        "prediction":   LABEL_NAMES[pred_idx],
        "confidence":   float(probs[pred_idx]),
        "prob_real":    float(probs[0]),
        "prob_fake":    float(probs[1]),
        "prob_satire":  float(probs[2]),
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fake news detection inference")
    parser.add_argument("--text",       type=str, default=None)
    parser.add_argument("--lang",       type=str, default=None,
                        help="Language hint: en | ru | uz")
    parser.add_argument("--csv",        type=str, default=None)
    parser.add_argument("--output",     type=str, default="results/predictions.csv")
    parser.add_argument("--model_dir",  type=str, default="models/best_model")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer, device = load_model(args.model_dir)

    if args.text:
        result = predict_text(args.text, model, tokenizer, device)
        lang_display = LANG_NAMES.get(args.lang, "Unknown") if args.lang else "Auto"

        print(f"\n{'='*48}")
        print(f"  Text:       {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
        print(f"  Language:   {lang_display}")
        print(f"  Prediction: {LABEL_EMOJI[LABEL_NAMES.index(result['prediction'])]} {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  ─────────────────────────────────")
        print(f"  Real:    {result['prob_real']:.1%}")
        print(f"  Fake:    {result['prob_fake']:.1%}")
        print(f"  Satire:  {result['prob_satire']:.1%}")
        print(f"{'='*48}\n")

    elif args.csv:
        df = pd.read_csv(args.csv)
        assert "text" in df.columns, "CSV must have a 'text' column"

        results = []
        for _, row in df.iterrows():
            result = predict_text(str(row["text"]), model, tokenizer, device)
            results.append(result)
            emoji = LABEL_EMOJI[LABEL_NAMES.index(result["prediction"])]
            print(f"  {emoji} {result['prediction']:<8} ({result['confidence']:.1%})  |  {str(row['text'])[:60]}")

        df_out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        os.makedirs(Path(args.output).parent, exist_ok=True)
        df_out.to_csv(args.output, index=False)
        print(f"\n[Results] Saved to {args.output}")

        # Summary
        preds = [r["prediction"] for r in results]
        for label in LABEL_NAMES:
            count = preds.count(label)
            print(f"  {LABEL_EMOJI[LABEL_NAMES.index(label)]} {label}: {count} ({count/len(preds):.1%})")

    else:
        print("[Error] Provide --text or --csv")


if __name__ == "__main__":
    main()
