# mistral-7b-docstring

Fine-tuning Mistral 7B with QLoRA on Python docstring generation — outperforming Llama 3.3 70B (a model 10x larger) on ROUGE-L and BERTScore at a fraction of the inference cost.

> **Resume bullet:** Fine-tuned Mistral 7B using QLoRA on 8,000 Python functions, outperforming Llama 3.3 70B on domain-specific NumPy-style docstring generation across ROUGE-L and BERTScore metrics at 95% lower inference cost.

---

## Results

| Model | ROUGE-L | BERTScore F1 |
|---|---|---|
| **Mistral 7B fine-tuned (ours)** | **0.2033** | **0.7739** |
| Llama 3.3 70B via Groq | 0.1715 | 0.7594 |
| Mistral 7B base (no fine-tuning) | 0.1102 | 0.7118 |

Evaluated on 100 held-out Python functions from CodeSearchNet never seen during training.

---

## Model

The fine-tuned model is hosted on HuggingFace Hub:
**[kk014/mistral-7b-docstring](https://huggingface.co/kk014/mistral-7b-docstring)**

---

## Project structure

```
mistral-docstring/
  data/
    collect_data.py       # scrapes and cleans CodeSearchNet
    train.jsonl           # 8,000 training samples
    val.jsonl             # 1,000 validation samples
    test.jsonl            # 500 held-out eval samples
  train/
    train.ipynb           # QLoRA fine-tuning notebook (Kaggle T4 x2)
  eval/
    eval.ipynb            # evaluation notebook (Google Colab)
    results.json          # raw scores for all three models
    summary.txt           # markdown table
  README.md
  requirements.txt
```

---

## Quickstart

### 1. Clone and set up

```bash
git clone https://github.com/kk014/mistral-docstring
cd mistral-docstring
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Collect data

```bash
python data/collect_data.py
```

Downloads CodeSearchNet from HuggingFace, parses Python functions and docstrings using AST, filters low-quality samples, and saves train/val/test splits to `./data/`.

### 3. Fine-tune

Open `train/train.ipynb` on [Kaggle](https://kaggle.com) with a T4 GPU accelerator attached.

- Upload `./data/` as a Kaggle dataset
- Add your HuggingFace token as a Kaggle secret (`HF_TOKEN`)
- Run all cells — training takes ~4 hours on T4 x2

The notebook uses QLoRA (4-bit NF4 quantisation + LoRA rank 16) via HuggingFace PEFT and TRL. Adapter weights are pushed to HuggingFace Hub automatically every 200 steps.

### 4. Evaluate

Open `eval/eval.ipynb` on [Google Colab](https://colab.research.google.com) with a T4 GPU.

- Upload `data/test.jsonl` when prompted
- Add your Groq API key to Colab Secrets (`GROQ_KEY`)
- Run all cells — evaluation takes ~1 hour

Computes ROUGE-L and BERTScore F1 for all three models and saves results to `eval/results.json`.

---

## Method

### Why QLoRA?

Full fine-tuning of a 7B model requires ~112GB VRAM. QLoRA reduces this to ~6GB by:
- Loading the base model in 4-bit NF4 quantisation (reduces model size by 4x)
- Training only small LoRA adapter matrices (~0.5% of total parameters)
- Using paged optimisers to handle memory spikes

This makes fine-tuning a 7B model possible on a single free T4 GPU.

### Data pipeline

```
CodeSearchNet (HuggingFace) → AST parser → quality filter → JSONL
```

Raw functions are parsed with Python's `ast` module to extract the function body (without docstring) as input and the docstring as the target output. Low-quality samples (too short, placeholder text, no punctuation) are filtered out.

### Training setup

```
Base model  : mistralai/Mistral-7B-v0.1
Method      : QLoRA (NF4 4-bit + LoRA rank 16, alpha 32)
Dataset     : 8,000 training samples from CodeSearchNet
Hardware    : Kaggle T4 x2 (free tier)
Time        : ~4 hours (1 epoch)
Framework   : HuggingFace PEFT + TRL SFTTrainer
```

### Evaluation

Both ROUGE-L and BERTScore are computed on 100 held-out test samples:

- **ROUGE-L** measures longest common subsequence overlap between generated and reference docstring
- **BERTScore F1** measures semantic similarity using distilbert-base-uncased embeddings

---

## Tech stack

| Tool | Purpose |
|---|---|
| HuggingFace Transformers | Model loading and inference |
| PEFT | LoRA adapter training |
| TRL SFTTrainer | Supervised fine-tuning loop |
| bitsandbytes | 4-bit quantisation |
| Kaggle T4 x2 | Free GPU for training |
| Groq API | Llama 3.3 70B inference for comparison |
| evaluate + bert-score | ROUGE-L and BERTScore computation |

---

## Requirements

```
# Data collection (laptop)
datasets==2.19.0
tqdm==4.66.4

# Evaluation (laptop or Colab)
evaluate==0.4.3
rouge-score==0.1.2
bert-score==0.3.13
transformers==4.46.0
peft==0.11.1
bitsandbytes==0.45.5
accelerate==0.34.0
groq
```

Training dependencies are installed inside the Kaggle notebook directly.

---

## License

Apache 2.0 — same as the base Mistral 7B model.