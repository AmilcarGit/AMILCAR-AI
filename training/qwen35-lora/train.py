import json
import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = os.getenv("AMI_BASE_MODEL", "unsloth/Qwen3.5-4B")
DATA_FILE = os.getenv("AMI_TRAIN_FILE", "data/train.jsonl")
OUTPUT_DIR = os.getenv("AMI_OUTPUT_DIR", "outputs/ami-qwen35")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora")
MAX_SEQ_LENGTH = int(os.getenv("AMI_MAX_SEQ_LENGTH", "2048"))
SEED = 3407

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en {path}:{line_no}: {exc}") from exc
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_no}: se requiere messages con al menos user y assistant")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{line_no}: el último mensaje debe ser assistant")
            if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in messages):
                raise ValueError(f"{path}:{line_no}: cada mensaje debe tener content no vacío")
            rows.append({"messages": messages})
    if len(rows) < 20:
        raise ValueError("Usa al menos 20 conversaciones; 300–1000 revisadas es un objetivo mejor.")
    return Dataset.from_list(rows)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere una GPU NVIDIA CUDA para este entrenamiento.")
    print("GPU:", torch.cuda.get_device_name(0))

    dataset = read_jsonl(DATA_FILE)
    split = dataset.train_test_split(test_size=0.10, seed=SEED)

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
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    def render(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_data = split["train"].map(render)
    eval_data = split["test"].map(render)

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        packing=False,
        dataset_num_proc=1,
        seed=SEED,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=args,
    )

    result = trainer.train()
    print("Métricas de entrenamiento:", result.metrics)
    print("Evaluación:", trainer.evaluate())

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("Adaptador LoRA guardado en:", ADAPTER_DIR)

    if os.getenv("AMI_EXPORT_GGUF", "0") == "1":
        gguf_dir = os.path.join(OUTPUT_DIR, "gguf")
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method=os.getenv("AMI_GGUF_QUANT", "q4_k_m"),
        )
        print("GGUF exportado en:", gguf_dir)

if __name__ == "__main__":
    main()
