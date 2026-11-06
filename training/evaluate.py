#!/usr/bin/env python3
"""Prueba rápida del adaptador LoRA entrenado de AMI."""
import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel

MODEL_DIR = os.getenv("AMI_ADAPTER_DIR", "outputs/ami-qwen35/lora-adapter")
EVAL_FILE = os.getenv("AMI_EVAL_FILE", "training/data/eval.jsonl")
SYSTEM = """Eres AMI, un asistente en español neutro. Responde de forma natural, clara,
directa y útil. No inventes datos. Distingue hechos de hipótesis y admite cuando no sabes."""

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("La evaluación de este script requiere una GPU CUDA.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    FastLanguageModel.for_inference(model)
    data = load_dataset("json", data_files=EVAL_FILE, split="train")
    for i, row in enumerate(data, 1):
        messages = row["messages"]
        if not messages or messages[0]["role"] != "user":
            continue
        prompt_messages = [{"role": "system", "content": SYSTEM}] + messages[:-1]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                repetition_penalty=1.08,
            )
        answer = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print(f"\n=== CASO {i} ===")
        print("PREGUNTA:", messages[-2]["content"] if len(messages) > 1 else messages[0]["content"])
        print("REFERENCIA:", messages[-1]["content"])
        print("AMI:", answer.strip())
        print("Revisa factualidad, utilidad, naturalidad y honestidad manualmente.")

if __name__ == "__main__":
    main()
