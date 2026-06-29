from transformers import AutoTokenizer
from config import MODEL_NAME, BLOCK_SIZE
from datasets import Dataset

"""
Токенизация и формирование блоков для сшивания текстов
"""

def load_tokenizer():
    """
    Возвращает токенизатор
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def tokenize_batch(batch):
    """
    Превращает текст в токены
    """
    tokenizer = load_tokenizer()
    return tokenizer(batch["text"])


def group_tokens_into_blocks(examples):
    """
    Сшивает токены в один поток и делит на равные блоки по 2048
    """

    concatenated = {}

    # Перебираем ключи датасета
    for k in examples.keys():
        flat_list = []  # Сюда будем собирать все токены в одну длинную линию

        # Перебираем каждый отдельный текст в этой колонке
        for sublist in examples[k]:
            flat_list.extend(sublist)

        concatenated[k] = flat_list

    # Считаем длину, кратную размеру блока
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE


    # Словарь для финального результата
    result = {}

    # NOTE: токены в конце строки отбрасываются
    # Нарезаем на куски строго по BLOCK_SIZE
    for k, t in concatenated.items():
        chunks = []

        for i in range(0, total_length, BLOCK_SIZE):
            chunk = t[i: i + BLOCK_SIZE]
            chunks.append(chunk)

        result[k] = chunks

    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_dataset(dataset: Dataset) -> Dataset:
    """
    Запуск пайплайн обработки данных
    """

    # Запускаем первичную токенизацию
    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

    # Запускаем группировку в блоки
    lm_dataset = tokenized.map(group_tokens_into_blocks, batched=True)

    return lm_dataset