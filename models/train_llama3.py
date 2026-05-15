"""
QLoRA Fine-Tuning: LLaMA 3.2-3B-Instruct on Indian Recipe Dataset

Technique: QLoRA = 4-bit NF4 quantization (base model frozen) + LoRA adapters (trainable).
Only ~10M of 3B parameters are trained. Runs on a single Kaggle T4 GPU (16GB VRAM).

Key design decisions (see docs/phases/phase3_fine_tuning/step1_qlora_concepts.md):
  - SFTTrainer with assistant_only_loss=True: loss computed on assistant response only
  - Conversational dataset format (messages column): trl 1.x native format
  - 7 LoRA target modules (all attention + MLP) for better vocabulary adaptation
  - paged_adamw_8bit: Adam optimizer states paged to CPU — saves ~0.5GB VRAM
  - device_map="cuda:0": force single GPU — "auto" splits across GPUs causing overhead

Library versions used:  trl>=1.0  transformers>=4.45  peft>=0.13  bitsandbytes>=0.46

Run from project root on Kaggle:
    python models/train_llama3.py

Dry-run (no GPU — syntax + config check only):
    uv run python models/train_llama3.py --dry-run
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.base_utils.common_utils import read_params

# ─────────────────────────────────────────────────────────────────────────────
# Constants — all tuneable values in one place
# ─────────────────────────────────────────────────────────────────────────────

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # All attention projections + all MLP projections.
    # Targeting 7 modules (vs the 2-module minimum) improves vocabulary adaptation
    # for domain-specific fine-tuning without meaningfully increasing VRAM.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

TRAIN_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,       # effective batch size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_seq_length=1024,
    optim="paged_adamw_8bit",            # pages optimizer states to CPU — saves ~0.5GB
    gradient_checkpointing=True,         # trades compute for memory on T4
    bf16=True,                           # bfloat16 compute (T4 supports bf16)
    fp16=False,
    logging_steps=50,
    eval_strategy="epoch",               # evaluate on val.csv after each epoch
    save_strategy="epoch",
    save_total_limit=1,                  # keep only the best checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="mlflow",
    # trl 1.x: assistant_only_loss=True masks all non-assistant tokens (label=-100)
    # so loss is only computed on the recipe response, not on the prompt
    assistant_only_loss=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model + Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(model_id: str) -> tuple:
    """
    Load LLaMA 3.2-3B-Instruct in 4-bit NF4 quantization and attach LoRA adapters.

    BitsAndBytesConfig:
      load_in_4bit=True         — store weights as 4-bit integers (8× memory reduction)
      bnb_4bit_quant_type="nf4" — NormalFloat4: 16 values spaced to match weight distribution
      bnb_4bit_compute_dtype=bfloat16 — dequantize to bf16 for actual matrix multiplications
      bnb_4bit_use_double_quant=True  — also quantize the quantization constants (~0.3GB saved)

    After quantization: ~2.5GB VRAM for a 3B model (vs ~12GB at float32).

    prepare_model_for_kbit_training:
      - Upcasts layer norms to float32 (they need full precision even in a 4-bit model)
      - Enables input gradient tracking so the LoRA backward pass can flow through the
        frozen 4-bit base model
      - Without this step: training fails or produces NaN losses
    """
    gc.collect()
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # right-padding: pad tokens go at the end — important for causal LM training
    # (left-padding would put pad tokens before the prompt, confusing the model)
    tokenizer.padding_side = "right"

    print(f"Loading model in 4-bit NF4: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda:0",     # force single GPU — "auto" splits across GPUs causing overhead
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,         # must be False when gradient_checkpointing=True
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset Formatting
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_format_dataset(
    train_path: str,
    val_path: str,
) -> tuple[Dataset, Dataset]:
    """
    Load train.csv and val.csv. Format each row as a 'messages' list of role/content dicts.

    trl 1.x SFTTrainer with assistant_only_loss=True expects a 'messages' column where
    each row is: [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]

    SFTTrainer then:
      1. Applies the tokenizer's chat template to convert messages → token IDs
      2. Identifies the assistant turn boundaries
      3. Sets labels = -100 for all non-assistant tokens (system + user turns)
      4. Computes cross-entropy loss only on the recipe response tokens
    """
    print(f"Loading training data:   {train_path}")
    train_df = pd.read_csv(train_path, usecols=["system_prompt", "user_message", "assistant_response"])
    print(f"Loading validation data: {val_path}")
    val_df = pd.read_csv(val_path, usecols=["system_prompt", "user_message", "assistant_response"])

    print(f"  Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    def _to_messages(row: pd.Series) -> list[dict]:
        return [
            {"role": "system",    "content": row["system_prompt"]},
            {"role": "user",      "content": row["user_message"]},
            {"role": "assistant", "content": row["assistant_response"]},
        ]

    train_messages = [_to_messages(row) for _, row in train_df.iterrows()]
    val_messages   = [_to_messages(row) for _, row in val_df.iterrows()]

    return (
        Dataset.from_dict({"messages": train_messages}),
        Dataset.from_dict({"messages": val_messages}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trainer
# ─────────────────────────────────────────────────────────────────────────────

def _build_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
) -> SFTTrainer:
    """
    Build the SFTTrainer.

    assistant_only_loss=True is the trl 1.x way of doing loss masking for conversational
    datasets. It replaces the older DataCollatorForCompletionOnlyLM approach.
    It automatically handles the masking for any chat-template model (LLaMA, Mistral, etc.)
    without needing to specify a response template string manually.
    """
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=TRAIN_CONFIG["max_seq_length"],
        **{k: v for k, v in TRAIN_CONFIG.items() if k != "max_seq_length"},
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(params: dict, hf_token: str | None = None, push_to_hub: bool = True) -> str:
    """
    Full training pipeline:
      1. HuggingFace login (for gated LLaMA model access)
      2. Load model + tokenizer in 4-bit with LoRA adapters attached
      3. Format train.csv + val.csv as messages-column datasets
      4. Build SFTTrainer with assistant_only_loss=True
      5. Train for 3 epochs, evaluate on val.csv after each epoch
      6. Save checkpoint locally + push LoRA adapter to HuggingFace Hub
    Returns the path to the saved model directory.
    """
    if hf_token:
        login(token=hf_token)
        print("HuggingFace authentication: OK")

    model_id   = params["model_candidates"]["llama3_3b"]
    train_path = params["data_dir"]["train"]
    val_path   = params["data_dir"]["val"]
    # Allow Kaggle (read-only dataset mount) to redirect output to /kaggle/working/
    output_dir = os.environ.get("MODEL_OUTPUT_DIR", params["model_output"]["output_dir"])

    os.environ["MLFLOW_EXPERIMENT_NAME"] = "qlora-indian-recipe-finetune"
    mlflow.set_experiment("qlora-indian-recipe-finetune")

    model, tokenizer = _load_model_and_tokenizer(model_id)
    train_dataset, val_dataset = _load_and_format_dataset(train_path, val_path)
    trainer = _build_trainer(model, tokenizer, train_dataset, val_dataset, output_dir)

    print("\nStarting QLoRA training...")
    print(f"  Model:            {model_id}")
    print(f"  Train examples:   {len(train_dataset)}")
    print(f"  Val examples:     {len(val_dataset)}")
    print(f"  Epochs:           {TRAIN_CONFIG['num_train_epochs']}")
    print(f"  Effective batch:  {TRAIN_CONFIG['per_device_train_batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps']}")
    print(f"  Learning rate:    {TRAIN_CONFIG['learning_rate']}")
    print(f"  LoRA rank:        {LORA_CONFIG['r']}  alpha: {LORA_CONFIG['lora_alpha']}")

    trainer.train()

    final_dir = str(Path(output_dir) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved locally: {final_dir}")

    if push_to_hub:
        hub_repo = "taljindergill78/indian-recipe-llama3.2-qlora"
        print(f"Pushing LoRA adapter to HuggingFace Hub: {hub_repo}")
        trainer.push_to_hub(hub_repo)
        mlflow.set_tag("hf_hub_repo", f"huggingface.co/{hub_repo}")
        print(f"Adapter available at: huggingface.co/{hub_repo}")

    return final_dir


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cli = argparse.ArgumentParser(
        description="QLoRA fine-tuning for LLaMA 3.2-3B-Instruct on Indian recipes.",
    )
    cli.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config only — do not load model or start training.",
    )
    cli.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing the LoRA adapter to HuggingFace Hub after training.",
    )
    cli.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning_rate (default: 2e-4). E.g. --lr 1e-4",
    )
    cli.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="Override LoRA rank r (default: 16). E.g. --lora-r 32",
    )
    args = cli.parse_args()

    if args.lr is not None:
        TRAIN_CONFIG["learning_rate"] = args.lr
        print(f"[CLI override] learning_rate = {args.lr}")
    if args.lora_r is not None:
        LORA_CONFIG["r"] = args.lora_r
        LORA_CONFIG["lora_alpha"] = args.lora_r * 2   # keep alpha = 2×r convention
        print(f"[CLI override] LoRA rank r = {args.lora_r}, alpha = {args.lora_r * 2}")

    params = read_params("params.yaml")

    if args.dry_run:
        print("=== DRY RUN — config only ===")
        print(f"  Model:        {params['model_candidates']['llama3_3b']}")
        print(f"  Train data:   {params['data_dir']['train']}")
        print(f"  Val data:     {params['data_dir']['val']}")
        effective_output = os.environ.get("MODEL_OUTPUT_DIR", params["model_output"]["output_dir"])
        print(f"  Output dir:   {effective_output}")
        print(f"  LoRA rank:    {LORA_CONFIG['r']}")
        print(f"  LoRA alpha:   {LORA_CONFIG['lora_alpha']}")
        print(f"  LoRA modules: {LORA_CONFIG['target_modules']}")
        print(f"  Epochs:       {TRAIN_CONFIG['num_train_epochs']}")
        print(f"  LR:           {TRAIN_CONFIG['learning_rate']}")
        print(f"  Max seq len:  {TRAIN_CONFIG['max_seq_length']}")
        return

    # HF token: read from environment variable
    # On Kaggle: set via UserSecretsClient (see Kaggle notebook setup cell)
    # Locally:   export HF_TOKEN=hf_xxx...
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set. Cannot download gated LLaMA model.")
        print("  On Kaggle: run UserSecretsClient cell first.")
        print("  Locally:   export HF_TOKEN=your_token_here")

    train(params, hf_token=hf_token, push_to_hub=not args.no_push)


if __name__ == "__main__":
    main()
