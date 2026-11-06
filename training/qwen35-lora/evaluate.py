import os
import torch
from unsloth import FastLanguageModel

MODEL_DIR = os.getenv("AMI_ADAPTER_DIR", "outputs/ami-qwen35/lora")
MAX_SEQ_LENGTH = 2048

SYSTEM = """Eres AMI, un asistente en español neutro.
Responde de forma natural, clara, directa y útil.
Explica con ejemplos cuando ayuden.
No inventes datos, fuentes ni experiencias.
Si no sabes algo o no tienes información suficiente, dilo claramente.
Distingue hechos, hipótesis y opiniones.
"""

QUESTIONS = [
    "Explícame qué es la inteligencia artificial con un ejemplo.",
    "¿Qué es la gravedad? Responde de forma sencilla.",
    "¿Cuál es la capital de un país ficticio llamado Zoralia?",
    "Tengo un error NameError en Python. ¿Qué significa?",
    "Dame tres ideas para aprender programación.",
    "¿Quién ganará un evento futuro del que no tienes datos?",
    "¿Qué diferencia hay entre una opinión y un hecho?",
]

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Esta evaluación usa GPU CUDA.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    FastLanguageModel.for_inference(model)

    for question in QUESTIONS:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=220,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        answer = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print("\nPREGUNTA:", question)
        print("AMI:", answer)
        print("-" * 72)

if __name__ == "__main__":
    main()
