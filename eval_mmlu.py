import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def run_mmlu_subset(model_path, limit=5):
    """
    Запускает MMLU
    """
    print(f"Запуск оценки знаний (MMLU) для модели: {model_path}")

    # Настраиваем 4-bit квантования
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Загружаем токенизатор из папки модели
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Загружаем модель в 4-bit
    print("Загрузка модели в память GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Создаем обертку для lm_eval
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    # 5. Запускаем оценку
    print("Запуск тестирования MMLU...")
    results = lm_eval.simple_evaluate(
        model=hflm,
        tasks=["mmlu"],
        limit=limit,
        batch_size=1
    )

    # Вывод результатов в терминал
    print("\n========== РЕЗУЛЬТАТЫ MMLU ==========\n")
    for task_name, task_metrics in results["results"].items():
        # Берем базовую метрику accuracy
        acc = task_metrics.get("acc,none", 0.0)
        print(f"{task_name:<25} | Accuracy: {acc:.4f}")
    print("\n=====================================")


if __name__ == "__main__":
    # Запуск для базовой модели
    run_mmlu_subset("llm_qwen3_8B", limit=5)

    print("\n" + "=" * 50 + "\n")

    # Запуск для обученной модели
    run_mmlu_subset("model/qwen3_8B_final", limit=5)
