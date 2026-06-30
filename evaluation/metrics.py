import re
import json
import numpy as np
from rapidfuzz.distance import Levenshtein
import sacrebleu


def normalize(text):
    """
    Нормализация текста для корректного сравнения
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def levenshtein_similarity(pred, ref):
    """
    Расстояние Левенштейна (определение сходства двух строк)
    """
    if len(ref) == 0:
        return 0.0
    dist = Levenshtein.distance(normalize(pred), normalize(ref))
    max_len = max(len(pred), len(ref))
    # Нормализуем расстояние (метрика от 0 до 1)
    return 1 - dist / max_len if max_len > 0 else 0.0


def chrf_score(preds, refs):
    """
    ChrF++ для оценки качества машинного перевода (для русского языка)
    """
    return sacrebleu.corpus_chrf(preds, [refs]).score


def bleu_score(preds, refs):
    """
    Классическая метрика BLEU оценки качества машинного перевода
    """
    return sacrebleu.corpus_bleu(preds, [refs]).score


def compute_basic_metrics(samples):
    """
    Агрегация результатов
    """
    preds = [s["prediction"] for s in samples]
    refs = [s["reference"] for s in samples]

    lev = []

    for p, r in zip(preds, refs):
        lev.append(levenshtein_similarity(p, r))

    return {
        "levenshtein": float(np.mean(lev)),
        "chrf": chrf_score(preds, refs),
        "bleu": bleu_score(preds, refs)
    }


def print_metrics_table(base_metrics, ft_metrics):
    """
    Вывод результатов в консоль
    """
    print("\n========== METRICS COMPARISON ==========\n")

    keys = base_metrics.keys()

    print(f"{'Metric':<20}{'Base':<15}{'Fine-tuned':<15}{'Delta':<15}")
    print("-" * 65)

    for k in keys:
        b = base_metrics[k]
        f = ft_metrics[k]
        d = f - b
        print(f"{k:<20}{b:<15.4f}{f:<15.4f}{d:<15.4f}")


def load_jsonl(path):
    """
    Загрузка результатов в JSON
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def evaluate(base_path, ft_path):
    """
    Запуск вычислений
    """
    base_samples = load_jsonl(base_path)
    ft_samples = load_jsonl(ft_path)

    base_metrics = compute_basic_metrics(base_samples)
    ft_metrics = compute_basic_metrics(ft_samples)

    print_metrics_table(base_metrics, ft_metrics)

    return {
        "base": base_metrics,
        "fine_tuned": ft_metrics
    }