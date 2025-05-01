# 🧾 Fine-Tuning LLaMA 3.2 for Clause Extraction on Legal Contracts

This project fine-tunes Meta’s open-weight **LLaMA 3.2 1B Instruct** model on the [CUAD dataset](https://arxiv.org/abs/2103.06268) for **clause extraction** in legal contracts using **QLoRA** and **parameter-efficient fine-tuning (PEFT)** with `TRL`'s `SFTTrainer`.

🔍 The goal is to extract precise clause spans (e.g., indemnity, change of control) from complex contract texts, suitable for deployment on low-resource, on-premise hardware while maintaining high accuracy.

---



## 📖 Paper Summary

> *Ritik Agrawal and Dr. Anru Zhang, 2025*  
> *“Generative AI for Legal Documents”*

- **Problem**: Manual clause extraction is slow, error-prone, and prior static models perform poorly.
- **Solution**: Fine-tune small open-source LLMs (LLaMA 3.2 1B) using LoRA on CUAD for accurate span-level clause retrieval.
- **Results**: The LoRA-tuned model outperforms larger proprietary models like Gemini 2.5 and OpenAI GPT-4o-mini on Exact Match and Levenshtein similarity.

---

## 📊 Key Results

| Model                    | EM ↑  | F1 ↑  | ROUGE-L ↑ | LevSim ↑ |
|-------------------------|-------|-------|-----------|----------|
| GPT-4.1 nano (OpenAI)   | 20.00 | 53.06 | 0.5196    | 0.4748   |
| GPT-4o-mini (OpenAI)    | 30.00 | 57.59 | 0.5570    | 0.5343   |
| Gemini 2.0 Flash Lite   | 41.94 | 60.63 | **0.5934**| 0.5571   |
| **LLaMA 3.2 1B + LoRA** ✅ | **46.00** | 56.40 | 0.5610    | **0.5925** |