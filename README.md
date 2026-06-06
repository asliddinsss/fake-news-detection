# Multilingual Fake News Detection
### Detecting Misinformation in Uzbek, Russian & English using Fine-tuned Transformers

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Series](https://img.shields.io/badge/Asliddin%20Builds-%2302-purple?style=flat-square)

> **Real-world impact:** Misinformation spreads faster in languages with fewer AI safety tools. Uzbek is spoken by 35+ million people yet has almost no public NLP infrastructure for fake news detection. This project builds a 3-class multilingual classifier (Real / Fake / Satire) covering Uzbek, Russian, and English — one of the first public models targeting Central Asian misinformation.

---

## Problem Statement

Fake news is not just an English-language problem. In Central Asia, misinformation spreads rapidly across Telegram channels, local news sites, and social media — largely unchecked because existing detection tools are built for English only.

This project fine-tunes **XLM-RoBERTa** (a multilingual transformer pre-trained on 100 languages including Uzbek and Russian) to classify news articles and social media posts into three categories:

| Label | Description |
|---|---|
| `0 — Real` | Factually accurate, sourced reporting |
| `1 — Fake` | Deliberately false or fabricated content |
| `2 — Satire` | Intentionally humorous/fictional, often misread as real |

---

## Results

| Model | Language | Accuracy | Macro F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | EN only | 78.3% | 0.76 |
| mBERT fine-tuned | Multilingual | 84.1% | 0.83 |
| XLM-RoBERTa fine-tuned | Multilingual | **91.2%** | **0.90** |
| XLM-RoBERTa + data augmentation | Multilingual | **92.8%** | **0.91** |

**Per-language breakdown (XLM-RoBERTa):**

| Language | Accuracy | F1 |
|---|---|---|
| English | 94.1% | 0.93 |
| Russian | 91.8% | 0.90 |
| Uzbek | 88.3% | 0.87 |

**Key finding:** Uzbek performance lags behind English by ~6% — primarily due to limited Uzbek training data. This gap narrows significantly with back-translation augmentation (Russian → Uzbek), improving Uzbek F1 from 0.81 to 0.87.

---

## 🗂️ Repository Structure

```
fake-news-detection/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb        # EDA across all three languages
│   ├── 02_baseline_tfidf.ipynb          # TF-IDF + classical ML baseline
│   └── 03_xlmroberta_finetune.ipynb     # XLM-RoBERTa fine-tuning + results
│
├── src/
│   ├── dataset.py                       # Dataset class + preprocessing pipeline
│   ├── model.py                         # Model definitions (baseline + transformer)
│   ├── train.py                         # Training loop with logging
│   ├── evaluate.py                      # Metrics, confusion matrix, per-language breakdown
│   └── predict.py                       # Inference on new text (CLI + function)
│
├── data/
│   ├── raw/                             # Original downloaded datasets
│   ├── processed/                       # Cleaned, tokenized, split data
│   └── samples/                         # Example articles for quick testing
│
├── models/
│   └── best_model/                      # Saved HuggingFace model checkpoint
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── per_language_breakdown.png
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## Datasets

This project combines three public datasets:

| Dataset | Language | Size | Source |
|---|---|---|---|
| LIAR | English | 12,836 | [Kaggle](https://www.kaggle.com/datasets/mdepak/fakenewsnet) |
| FakeNewsNet | English | 23,196 | [GitHub](https://github.com/KaiDMML/FakeNewsNet) |
| RuFake | Russian | 8,942 | [HuggingFace](https://huggingface.co/datasets) |
| Uzbek News Corpus (custom) | Uzbek | 2,100 | Scraped + manually labeled |

The Uzbek subset was scraped from local news outlets (kun.uz, gazeta.uz, daryo.uz) and labeled manually — making this one of the first labeled Uzbek fake news datasets made publicly available.

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/asliddinsss/fake-news-detection.git
cd fake-news-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run inference on your own text
```bash
python src/predict.py --text "Vazirlik yangi qonun loyihasini tasdiqladi" --lang uz
```

Output:
```
Text:       Vazirlik yangi qonun loyihasini tasdiqladi
Language:   Uzbek (uz)
Prediction: Real
Confidence: 91.3%
  Real:    91.3%
  Fake:     6.2%
  Satire:   2.5%
```

### 4. Train from scratch
```bash
python src/train.py --model xlmroberta --epochs 5 --batch_size 16
```

### 5. Explore notebooks
```bash
jupyter notebook notebooks/
```

---

## Model Architecture

We use **XLM-RoBERTa-base** (`xlm-roberta-base`) with a custom 3-class classification head:

```
XLM-RoBERTa-base (pretrained, 270M params)
    ↓
[CLS] token representation (768-dim)
    ↓
Dropout(0.3)
    ↓
FC(768 → 256) + GELU + Dropout(0.2)
    ↓
FC(256 → 3) + Softmax
```

**Why XLM-RoBERTa over mBERT?**
- Pre-trained on 2.5TB of text across 100 languages
- Significantly better on low-resource languages like Uzbek
- Consistent tokenization across scripts (Latin Uzbek, Cyrillic Russian)

---

## 📈 Training Details

| Parameter | Value |
|---|---|
| Base model | `xlm-roberta-base` |
| Optimizer | AdamW |
| Learning rate | 2e-5 (with linear warmup) |
| Warmup steps | 10% of total steps |
| Batch size | 16 |
| Epochs | 5 |
| Max sequence length | 256 tokens |
| Loss | CrossEntropyLoss (class-weighted) |
| Hardware | Google Colab (T4 GPU) |

---

## Real-World Impact

- **Telegram monitoring:** The model can be deployed as a bot to flag suspicious posts in Uzbek/Russian Telegram channels before they go viral
- **Newsroom assistance:** Local journalists can use it to quickly fact-check incoming wire reports
- **Digital literacy:** The public Uzbek dataset enables future researchers to build on this work

This is **Asliddin Builds #02**, part of an ongoing series of ML projects tackling real problems.  
← Previous: [#01 — Deforestation Detection](https://github.com/YOUR_USERNAME/deforestation-detection)

---

## Future Work

- [ ] Expand Uzbek dataset to 10,000+ samples
- [ ] Add claim-level fact checking (not just article-level)
- [ ] Deploy as a Telegram bot for real-time monitoring
- [ ] Fine-tune on Tajik and Kyrgyz for broader Central Asian coverage

---

## Author

**Asliddin** — Grade 9, Presidential School, Namangan, Uzbekistan
AI/ML Researcher | APIO Finalist 2025 | TEDx Speaker
[LinkedIn](#) · [GitHub](#) · [YouTube](#)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
