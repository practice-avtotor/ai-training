import json
import re


def remove_thinking(text: str) -> str:
    """
    Удаляет блоки <think>...</think>
    """
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )

    return text.strip()


def save_jsonl(records, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path):
    result = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result.append(json.loads(line))
    return result