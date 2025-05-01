#!/usr/bin/env python
"""
finetune_cuad_llama3.py

Fine-tune meta-llama/Llama-3.2-1B-Instruct on CUAD clause-extraction
using QLoRA (PEFT) + SFTTrainer (TRL).
"""
import os
os.environ["TORCH_DISTRIBUTED_DISABLE_DTENSOR"] = "1"
import torch
local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
import argparse, json, re, torch, pandas as pd
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from trl import SFTConfig

# ────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cuad_json", default="/home/users/raa75/genAI/legal/CUAD_v1/CUAD_v1.json")
    p.add_argument("--output_dir", default="./llama3_cuad_adapter")
    p.add_argument("--num_train_epochs", type=int, default=80)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────
# Prompt template (identical to evaluation script)
# ────────────────────────────────────────────────────────────────────
def build_prompt(context: str, question: str) -> str:
    return f"""Extract the exact clause from the contract text that answers the question below. Only return the clause itself. Do not add any explanations or extra text.

### Contract:
{context}

### Question:
{question}

### Answer:"""

# ────────────────────────────────────────────────────────────────────
# CUAD → HuggingFace Dataset in instruction-tuning format
# ────────────────────────────────────────────────────────────────────
def load_cuad(cuad_json: str) -> pd.DataFrame:
    raw = json.load(open(cuad_json))
    rows = []
    for doc in raw["data"]:
        for para in doc["paragraphs"]:
            ctxt = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible"):      # skip unanswerable
                    continue
                ans  = qa["answers"][0]
                rows.append(
                    dict(
                        id          = qa["id"],
                        prompt      = build_prompt(ctxt, qa["question"]),
                        answer_text = ans["text"],
                    )
                )
    return pd.DataFrame(rows)

def make_hf_dataset(df: pd.DataFrame) -> Dataset:
    # → single “text” field that contains prompt + ground-truth answer
    texts = (df["prompt"] + " " + df["answer_text"].str.strip()).tolist()
    return Dataset.from_dict({"text": texts, "id": df["id"].tolist()})

# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("🔹 Loading CUAD …")
    df_full = load_cuad(args.cuad_json)
    train_df, val_df = train_test_split(df_full, test_size=0.05, random_state=args.seed)
    ds_train, ds_val = make_hf_dataset(train_df), make_hf_dataset(val_df)
    print(f"✅ Train: {len(ds_train):,}  •  Val: {len(ds_val):,}")

    # ── Model & tokenizer in 4-bit ────────────────────────────────
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    bnb_cfg  = BitsAndBytesConfig(
        load_in_8bit=True,          # ← use 8-bit instead
        llm_int8_threshold=6.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": local_rank},   #  ← **one full copy per rank**
            trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── LoRA adapter config ───────────────────────────────────────
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training arguments ────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        seed=args.seed,
        max_length=2048,
        dataset_text_field="text",
    )

    # ── Trainer (TRL’s SFTTrainer masks prompt tokens for you) ────
    trainer = SFTTrainer(
        model               = model,
        args                = training_args,
        train_dataset       = ds_train,
        eval_dataset        = ds_val,
        peft_config         = lora_cfg,
        processing_class    = tokenizer,   # ★ this activates the DTensor-safety shim
    )
    print("Tokenizer loaded from:", trainer.tokenizer.name_or_path)
    print("pad_token_id:", trainer.tokenizer.pad_token_id)
    print("🚀 Starting fine-tune …")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"🎉 Done! Adapter saved to → {args.output_dir}")

if __name__ == "__main__":
    main()
