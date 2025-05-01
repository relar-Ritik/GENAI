import os
import re
import json
import pandas as pd
import numpy as np
import evaluate
import Levenshtein
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from google import genai
from peft import AutoPeftModelForCausalLM

# === Setup keys ===
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# === Load dataset ===
df = pd.read_json("cuad_test_5pct.json", lines=True)
squad_metric = evaluate.load("squad")
rouge = evaluate.load("rouge")

def build_prompt(context: str, question: str) -> str:
    return f"""Extract the exact clause from the contract text that answers the question below. Only return the clause itself. Do not add ANY explanations or extra text.

### Contract:
{context}

### Question:
{question}

### Answer:"""

# === Model config ===
models = {
    "llama-Instruct": {
        "type": "hf",
        "id": "meta-llama/Llama-3.2-1B-Instruct"
    },
    "llama": {
        "type": "hf",
        "id": "meta-llama/Llama-3.2-1B"
    },
    "openai_4o_mini": {
        "type": "openai",
        "id": "gpt-4o-mini"
    },
    "openai_4.1_nano": {
        "type": "openai",
        "id": "gpt-4.1-nano"
    },
    "gemini_2.0_flash_light": {
        "type": "gemini",
        "id": "gemini-2.0-flash-lite"
    },
    "gemini_2.5_pro": {
        "type": "gemini",
        "id": "gemini-2.5-pro-exp-03-25"
    },
    "llama3-cuad-finetuned": {
        "type": "finetuned",
        "base_id": "meta-llama/Llama-3.2-1B-instruct",
        "adapter_path": "./llama3_cuad_adapter"
    }
}

results = []

def evaluate_model(name, model_config):
    print(f"\n🔍 Evaluating: {name}")
    predictions, references, rouge_ls, lev_sims = [], [], [], []

    if model_config["type"] == "hf":
        tokenizer = AutoTokenizer.from_pretrained(model_config["id"], use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_config["id"],
            torch_dtype=torch.float16,
            device_map="auto"
        )

        def generate(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False
            )
            return tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    elif model_config["type"] == "finetuned":
        tokenizer = AutoTokenizer.from_pretrained(model_config["base_id"], use_fast=False)
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_config["adapter_path"],
            torch_dtype="auto",
            device_map="auto"
        )
        model.eval()

        def generate(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False
            )
            return tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    elif model_config["type"] == "openai":
        def generate(prompt: str) -> str:
            response = openai_client.chat.completions.create(
                model=model_config["id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()

    elif model_config["type"] == "gemini":
        def generate(prompt: str) -> str:
            return genai_client.generate_content(model=model_config["id"], contents=prompt).text.strip()

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"{name}"):
        prompt = build_prompt(row.context, row.question)
        pred_text = generate(prompt)
        true_text = row.answer_text.strip()

        pred_text = re.sub(r"^\W+|\W+$", "", pred_text)
        true_text = re.sub(r"^\W+|\W+$", "", true_text)

        predictions.append({"id": row.id, "prediction_text": pred_text})
        references.append({
            "id": row.id,
            "answers": {"text": [true_text], "answer_start": [row.answer_start]}
        })

        rouge_ls.append(
            rouge.compute(predictions=[pred_text], references=[true_text])["rougeL"]
        )
        max_len = max(len(pred_text), len(true_text))
        lev_sims.append(
            1.0 if max_len == 0 else
            1 - (Levenshtein.distance(pred_text, true_text) / max_len)
        )

    squad_results = squad_metric.compute(predictions=predictions, references=references)
    results.append({
        "Model": name,
        "Exact Match": squad_results["exact_match"],
        "F1 Score": squad_results["f1"],
        "ROUGE-L": np.mean(rouge_ls),
        "Levenshtein Sim": np.mean(lev_sims),
    })

# Run evaluations
for name, config in models.items():
    evaluate_model(name, config)

# === Report Results ===
results_df = pd.DataFrame(results)
print("\n Final Results:")
print(results_df.to_markdown(index=False))
