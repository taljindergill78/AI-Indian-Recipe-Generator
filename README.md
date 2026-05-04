# AI-Powered Indian Recipe Generator

## Project Overview

Fine-tune a large language model to generate culturally authentic Indian recipes from a list
of input ingredients. The user provides ingredients; the model generates a complete recipe
(name, instructions, cook time). A multi-turn adaptation layer then lets the user refine the
result — make it vegan, spicier, or quicker — using the same fine-tuned model in a second call.

Fine-tuning uses **QLoRA (Quantized Low-Rank Adaptation)** with 4-bit NF4 quantization,
which makes it possible to fine-tune a 3B parameter model on a free-tier Kaggle GPU (P100, 16GB VRAM).

## Dataset

- **Source**: [archanaskitchen.com](https://www.archanaskitchen.com/) via HuggingFace
  ([Anupam007/indian-recipe-dataset](https://huggingface.co/datasets/Anupam007/indian-recipe-dataset))
- **Total recipes**: 5,938
- **Indian recipes used for training** (after filtering by cuisine): ~3,792
- **Key columns**: `TranslatedRecipeName`, `TranslatedIngredients`, `Cleaned-Ingredients`,
  `TranslatedInstructions`, `TotalTimeInMins`, `Cuisine`, `image-url`

> **Data setup**: After cloning, run `uv run python scripts/download_data.py` to download
> the dataset into `data/raw/`.

## Models Evaluated

Three candidate models are evaluated at baseline (no fine-tuning) before selecting one to fine-tune:

| Model | Parameters | Why Considered |
|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Primary candidate — strong instruction following, fits on P100 with 4-bit quant |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Fastest to train — useful if 3B shows no improvement from fine-tuning |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Competitive quality at small size, strong baseline performance |

## Fine-Tuning Setup

- **Method**: QLoRA — LoRA adapters (rank 16) injected into Q/K/V/O attention projections
- **Quantization**: 4-bit NF4 (via `bitsandbytes`) — reduces GPU memory from ~12GB to ~5GB
- **Trainable parameters**: ~3M out of 3B (~0.1%)
- **Compute**: Kaggle Notebooks (NVIDIA P100, 16GB VRAM, free tier)
- **Tracking**: MLflow (hyperparameters + BLEU/ROUGE per epoch)

## Evaluation Metrics

| Metric | What It Measures |
|---|---|
| BLEU-4 | N-gram overlap between generated and reference instructions |
| ROUGE-L | Longest common subsequence — captures sentence-level fluency |
| Ingredient F1 | Precision + recall on ingredient names (harmonic mean) |
| BERTScore | Semantic similarity using contextual embeddings |

## Results

> ⚠️ **Metrics pending re-evaluation.** A bug was found and fixed in the BLEU implementation
> (the function was splitting on `'/n'` instead of `'\n'`, and using `sentence_bleu` instead
> of `corpus_bleu`). All previously reported numbers are invalid. Corrected baseline and
> fine-tuned metrics will be added here after Phase 2 evaluation is complete.

## Tech Stack

| Area | Tools |
|---|---|
| Fine-tuning | PyTorch, HuggingFace Transformers, PEFT, bitsandbytes, TRL |
| Data | HuggingFace Datasets, Pandas |
| Evaluation | NLTK (corpus_bleu), rouge-score, bert-score |
| Experiment tracking | MLflow |
| Frontend | Gradio Blocks (deployed on HuggingFace Spaces) |
| API | FastAPI + Uvicorn |
| Package management | UV (pyproject.toml + uv.lock) |

## Project Setup

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd AI-Indian-Recipe-Generator
uv sync

# 2. Download the dataset
uv run python scripts/download_data.py

# 3. Run tests
uv run pytest tests/
```
