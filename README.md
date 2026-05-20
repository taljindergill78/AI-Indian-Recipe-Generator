# AI-Powered Indian Recipe Generator

## Project Overview

Fine-tune a large language model to generate culturally authentic Indian recipes given a dish
name, diet type, and regional cuisine. A multi-turn adaptation layer then lets the user refine
the result — make it vegan, spicier, or quicker — using the same fine-tuned model in a second call.

Fine-tuning uses **QLoRA (Quantized Low-Rank Adaptation)** with 4-bit NF4 quantization, making
it possible to fine-tune a 3B parameter model on a single GPU with 0.75% of the total parameters trained.

**Model adapter on HuggingFace Hub**: [taljindergill78/indian-recipe-llama3.2-qlora](https://huggingface.co/taljindergill78/indian-recipe-llama3.2-qlora)

---

## Dataset

- **Source**: [archanaskitchen.com](https://www.archanaskitchen.com/) via HuggingFace
  ([Anupam007/indian-recipe-dataset](https://huggingface.co/datasets/Anupam007/indian-recipe-dataset))
- **Raw dataset**: 5,938 recipes (includes Continental, Italian, and other non-Indian cuisines)
- **After Indian cuisine filter**: 4,013 recipes (30+ regional Indian cuisines retained)
- **Training split**: 3,263 recipes | Validation: 250 | Test: 500 (held-out)
- **Key columns**: `RecipeName`, `TranslatedIngredients`, `Cleaned-Ingredients`, `TranslatedInstructions`, `TotalTimeInMins`, `Cuisine`, `Diet`

> **Data setup**: After cloning, run `uv run python scripts/download_data.py` to download the dataset into `data/raw/`.

---

## Model Selection

Three candidate models were evaluated at baseline (no fine-tuning) on a 500-recipe held-out
test set before selecting one to fine-tune:

| Model | Parameters | Why Considered |
|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` ✅ **selected** | 3B | Best baseline scores across all metrics |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Fast lower-bound comparison |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Strong small model; degenerate output on 58% of rows at baseline |

---

## Fine-Tuning

| Setting | Value |
|---|---|
| Method | QLoRA — 4-bit NF4 base (frozen) + LoRA adapters (trainable) |
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| LoRA rank / alpha | r=16, alpha=32 (scaling factor = 2.0) |
| LoRA target modules | 7 modules: q, k, v, o projections (attention) + gate, up, down (MLP) |
| Trainable parameters | 24,313,856 / 3,237,063,680 — **0.75%** of total |
| Epochs | 3 (eval_loss: 1.302 → 1.250 → 1.249) |
| Effective batch size | 16 (4 per device × 4 gradient accumulation) |
| Learning rate | 2e-4 with cosine decay + 5% warmup |
| Max sequence length | 1,024 tokens |
| Compute | NVIDIA A100-SXM4-80GB (RunPod Community Cloud) |
| Training time | 28 minutes 16 seconds (612 steps) |
| VRAM peak | 41 GB / 80 GB (52%) |
| Experiment tracking | MLflow — `qlora-indian-recipe-finetune` experiment |

---

## Evaluation Metrics

| Metric | What It Measures |
|---|---|
| Ingredient F1 | Token-level overlap between generated and reference ingredient lists (after fuzzy normalization) |
| ROUGE-L | Longest common subsequence overlap on cooking instructions |
| BERTScore F1 | Semantic similarity of instructions using RoBERTa embeddings |
| Corpus BLEU-4 | N-gram precision on instructions (strict surface-form match) |

All metrics computed on the 500-recipe held-out test set with 95% bootstrap confidence intervals.

---

## Results

### Baseline — Before Fine-Tuning (500 rows, greedy decoding)

| Model | Ingredient F1 | ROUGE-L | BERTScore F1 | Corpus BLEU |
|---|---|---|---|---|
| Phi-3-mini-4k (baseline) | 0.0188 | 0.0984 | 0.8018 | 0.0048 |
| LLaMA 3.2-1B (baseline) | 0.0255 | 0.1711 | 0.8372 | 0.0143 |
| LLaMA 3.2-3B (baseline) | 0.0308 | 0.1954 | 0.8455 | 0.0195 |

### After Fine-Tuning

| Model | Ingredient F1 | ROUGE-L | BERTScore F1 | Corpus BLEU |
|---|---|---|---|---|
| **LLaMA 3.2-3B fine-tuned** (this model) | **0.1881 ↑6.1×** | ⏳ v2 pending | ⏳ v2 pending | ⏳ v2 pending |

> **Ingredient F1 improved 6.1× at baseline**, confirming the model learned Indian ingredient
> vocabulary. ROUGE-L and BERTScore from the first evaluation run (v1) were invalid due to a
> greedy decoding repetition loop — the fine-tuned model's sharpened distributions caused
> ingredient repetition before reaching the instructions section. v2 re-evaluation uses nucleus
> sampling with `repetition_penalty=1.3` and is currently running.

---

## Tech Stack

| Area | Tools |
|---|---|
| Fine-tuning | PyTorch, HuggingFace Transformers, PEFT, bitsandbytes, TRL |
| Data | HuggingFace Datasets, Pandas |
| Evaluation | NLTK (corpus_bleu), rouge-score, bert-score, rapidfuzz |
| Experiment tracking | MLflow (training metrics) + HuggingFace Hub (model weights) |
| Frontend | Gradio Blocks — to be deployed on HuggingFace Spaces |
| API | FastAPI + Uvicorn |
| Package management | UV (pyproject.toml + uv.lock) |

---

## Quick Start

```bash
# 1. Clone and install dependencies
git clone https://github.com/taljindergill78/AI-Indian-Recipe-Generator.git
cd AI-Indian-Recipe-Generator
uv sync

# 2. Download the dataset
uv run python scripts/download_data.py

# 3. Run tests
uv run pytest tests/
```

### Generate a recipe using the fine-tuned model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base, "taljindergill78/indian-recipe-llama3.2-qlora")
tokenizer = AutoTokenizer.from_pretrained("taljindergill78/indian-recipe-llama3.2-qlora")

messages = [
    {"role": "system", "content": "You are an expert Indian chef. Generate authentic Indian recipes with detailed ingredients and clear step-by-step cooking instructions."},
    {"role": "user",   "content": "Generate a Vegetarian North Indian recipe for Dal Makhani"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=768, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Project Structure

```
AI-Indian-Recipe-Generator/
├── data/
│   ├── raw/                        ← downloaded datasets
│   └── processed/                  ← train.csv, val.csv, test.csv + eval outputs
├── models/
│   ├── train_llama3.py             ← QLoRA fine-tuning script
│   └── trained/                    ← local checkpoint (checkpoint-612/)
├── scripts/
│   ├── download_data.py
│   └── evaluation/
│       ├── evaluate.py             ← baseline evaluation (Phase 2)
│       ├── evaluate_finetuned.py   ← fine-tuned model evaluation (Phase 4)
│       ├── parser.py               ← structured output parser
│       └── metrics.py              ← ROUGE-L, BERTScore, BLEU, Ingredient F1
├── pyproject.toml                  ← UV dependencies
└── params.yaml                     ← paths and model config
```

---

## Links

- **HuggingFace Hub** (model adapter): [taljindergill78/indian-recipe-llama3.2-qlora](https://huggingface.co/taljindergill78/indian-recipe-llama3.2-qlora)
- **GitHub**: [taljindergill78/AI-Indian-Recipe-Generator](https://github.com/taljindergill78/AI-Indian-Recipe-Generator)
- **HuggingFace Spaces** (Gradio demo): coming in Phase 4
