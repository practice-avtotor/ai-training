import csv
import json
from config_eval import BASE_MODEL, FINETUNED_MODEL, OUTPUT_DIR, TEST_DATASET, BATCH_SIZE
from inference import InferenceEngine
from utils import load_jsonl, save_jsonl
from metrics import compute_basic_metrics


def run_inference(model_path, name):
    """
    Запуск модели и прогон через тестовый датасет
    """
    engine = InferenceEngine(model_path)
    samples = load_jsonl(TEST_DATASET)

    # Для тестирования порограммымы и отладки можно ограничится одним батчем
    # samples = samples[:BATCH_SIZE]

    results = []
    print(f"\nЗапуск модели: {name}\n")

    # Идем по датасету шагами размера BATCH_SIZE
    for i in range(0, len(samples), BATCH_SIZE):
        batch_samples = samples[i: i + BATCH_SIZE]

        # Подготавливаем промпты (все сообщения кроме последнего референса)
        batch_messages = [s["messages"][:-1] for s in batch_samples]
        batch_references = [s["messages"][-1]["content"] for s in batch_samples]

        # Генерируем ответы батчем
        predictions = engine.generate_batch(batch_messages)

        # Сохраняем результаты
        for j, pred in enumerate(predictions):
            results.append({
                "id": i + j,
                "reference": batch_references[j],
                "prediction": pred
            })

        print(f"{name}: Обработано {min(i + BATCH_SIZE, len(samples))} / {len(samples)}")

    output_path = OUTPUT_DIR / f"{name}_predictions.jsonl"
    save_jsonl(results, output_path)

    return results, output_path


def compute_all_metrics(samples):
    """
    Вычисляет все метрики
    """

    basic = compute_basic_metrics(samples)

    return {**basic} # , **term


def save_csv(report, path):
    """
    Сохраняет вывод в CSV
    """
    keys = list(report["base"].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["metric", "base", "fine_tuned", "delta"])

        for k in keys:
            base_v = report["base"][k]
            ft_v = report["fine_tuned"][k]
            delta = ft_v - base_v

            writer.writerow([k, base_v, ft_v, delta])


def print_table(report):
    """
    Печатает таблицу в терминал
    """
    print("\n========== FINAL COMPARISON ==========\n")

    print(f"{'Metric':<20}{'Base':<15}{'Fine-tuned':<15}{'Delta':<15}")
    print("-" * 65)

    for k in report["base"].keys():
        b = report["base"][k]
        f = report["fine_tuned"][k]
        d = f - b
        print(f"{k:<20}{b:<15.4f}{f:<15.4f}{d:<15.4f}")


def run():
    """
    Главный пайплайн
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    base_samples, _ = run_inference(BASE_MODEL, "base")
    ft_samples, _ = run_inference(FINETUNED_MODEL, "finetuned")

    base_metrics = compute_all_metrics(base_samples)
    ft_metrics = compute_all_metrics(ft_samples)

    report = {
        "base": base_metrics,
        "fine_tuned": ft_metrics
    }

    print_table(report)

    csv_path = OUTPUT_DIR / "comparison.csv"
    json_path = OUTPUT_DIR / "comparison.json"

    save_csv(report, csv_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- {csv_path}\n- {json_path}\n")


if __name__ == "__main__":
    run()