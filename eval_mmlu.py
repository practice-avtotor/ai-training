import subprocess
from config import SFT_OUTPUT_DIR


def run_mmlu_subset(model_path, limit=5):
    """
    Запускает урезанную версию MMLU для проверки базовых знаний
    """
    print(f"Запуск оценки знаний..")

    # Выбираем задачи MMLU
    tasks = "mmlu,ceval-hard,ru_mmlu,rudete"

    command = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},load_in_4bit=True,trust_remote_code=True",
        "--tasks", tasks,
        "--limit", str(limit),  # Берем только N вопросов
        "--device", "cuda:0",
        "--batch_size", "1"
    ]

    try:
        subprocess.run(command, check=True)
        print("Проверка успешно завершена!")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении lm_eval: {e}")


if __name__ == "__main__":
    # Запускаем после SFT
    run_mmlu_subset(SFT_OUTPUT_DIR)