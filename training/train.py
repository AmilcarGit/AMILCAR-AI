#!/usr/bin/env python3
"""Entrenamiento SFT + LoRA BF16 de AMI con Unsloth (Qwen3.5-4B)."""
import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

MODEL_NAME = os.getenv("AMI_BASE_MODEL", "unsloth/Qwen3.5-4B")
TRAIN_FILE = os.getenv("AMI_TRAIN_FILE", "training/data/train.jsonl")
EVAL_FILE = os.getenv("AMI_EVAL_FILE", "training/data/eval.jsonl")
OUTPUT_DIR = os.getenv("AMI_OUTPUT_DIR", "outputs/ami-qwen35")
MAX_SEQ_LENGTH = int(os.getenv("AMI_MAX_SEQ_LENGTH", "2048"))
EPOCHS = float(os.getenv("AMI_EPOCHS", "3"))
SEED = 3407

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No se detectó CUDA. Ejecuta este entrenamiento en una GPU NVIDIA (por ejemplo, Google Colab).")
    for path in (TRAIN_FILE, EVAL_FILE):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No existe {path}. Copia los JSONL de ejemplo y completa el dataset.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    raw = load_dataset("json", data_files={"train": TRAIN_FILE, "eval": EVAL_FILE})

    for split_name in ("train", "eval"):
        if len(raw[split_name]) == 0:
            raise ValueError(f"El archivo {split_name} está vacío.")

    def validate_and_format(example):
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("Cada línea debe tener messages con al menos user y assistant.")
        if messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
            raise ValueError("Cada ejemplo debe comenzar con user y terminar con assistant.")
        if any(m.get("role") not in ("user", "assistant", "system") for m in messages):
            raise ValueError("Rol no admitido; usa system, user o assistant.")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            raise ValueError("Todos los mensajes deben tener content de texto no vacío.")
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    train_ds = raw["train"].map(validate_and_format)
    eval_ds = raw["eval"].map(validate_and_format)
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        seed=SEED,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
    )
    trainer.train()

    adapter_dir = os.path.join(OUTPUT_DIR, "lora-adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adaptador LoRA guardado en: {adapter_dir}")

    gguf_dir = os.path.join(OUTPUT_DIR, "gguf")
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
    print(f"Exportación GGUF solicitada en: {gguf_dir}")
    print("Cuando termine, ejecuta: python training/create_modelfile.py")

if __name__ == "__main__":
    main()
