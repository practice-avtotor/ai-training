from pathlib import Path
from datasets import Dataset

"""
1) Предполагается, что все файлы для CPT в TXT формате
2) Собираем в структурированный датасет 
"""

def load_texts(folder: str):
    """
    Считывает все txt-файлы из директории
    """

    texts = []

    for file in Path(folder).rglob("*.txt"):
        try:
            text = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )
            text = text.strip()
            texts.append(text)
        except Exception as e:
            print(f"Ошибка чтения {file}: {e}")

    return texts


def build_dataset(folder: str):
    texts = load_texts(folder)
    return Dataset.from_dict({"text": texts})