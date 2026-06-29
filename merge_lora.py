import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL_NAME = "llm_qwen3_8B"
LORA_ADAPTER_DIR = "output/qwen_sft"
FINAL_OUTPUT_DIR = "model/qwen3_8B_final"
 
def merge_models():
    print("1. Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR)
 
    print("2. Загрузка базовой модели в bfloat16 (процесс на CPU/RAM)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Делаем всё в оперативной памяти, чтобы не переполнить VRAM видеокарты
        low_cpu_mem_usage=True
    )
 
    print("3. Наложение LoRA-адаптера на модель...")
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_DIR,
        torch_dtype=torch.bfloat16
    )

    print("4. Физическое слияние весов (Merge & Unload)...")
    # Слияние LoRa и модели
    merged_model = model_with_lora.merge_and_unload()
 
    print(f"5. Сохранение объединенной модели в: {FINAL_OUTPUT_DIR}...")
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    
    # Сохраняем результат
    merged_model.save_pretrained(FINAL_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)
 
    print("Слияние успешно завершено!")
 
if __name__ == "__main__":
    merge_models()
