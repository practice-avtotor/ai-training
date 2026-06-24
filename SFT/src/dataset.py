from datasets import load_dataset

def load_sft_dataset(path: str):
    """
    Загружает json с инструкциями
    """

    dataset = load_dataset("json", data_files=path, split="train")

    return dataset