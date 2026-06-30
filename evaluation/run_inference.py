import json
from pathlib import Path
from inference import InferenceEngine
from config_eval import *
from utils import save_jsonl


def run(model_path, output_name):
    # Загружаем модель
    engine = InferenceEngine(model_path)
    results = []

    # Генерируем ответы модели и сохраняем
    with open(TEST_DATASET, encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            sample = json.loads(line)
            messages = sample["messages"]
            prompt = messages[:-1]
            reference = messages[-1]["content"]
            prediction = engine.generate(prompt)

            print(f"{index+1}")

            results.append({
                "id": index,
                "reference": reference,
                "prediction": prediction,
                "prompt": prompt[-1]["content"]
            })

    output = OUTPUT_DIR / output_name
    save_jsonl(results, output)
    print(output)


if __name__ == "__main__":
    run(
        BASE_MODEL,
        "base_predictions.jsonl"
    )