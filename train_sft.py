from config import SFT_DATASET, SFT_OUTPUT_DIR, OUTPUT_DIR
from src.dataset import load_sft_dataset
from src.sft_trainer import create_sft_trainer
from src.model import load_cpt_model, load_model
from src.tokenizer import load_tokenizer
import gc
import torch

def main():
    # Очищаем VRAM от остатков прошлых падений скрипта
    gc.collect()
    torch.cuda.empty_cache()

    print("Загрузка датасета...")

    dataset = load_sft_dataset(SFT_DATASET)

    print(f"Примеров: {len(dataset)}")

    print("Загрузка токенизатора...")

    tokenizer = load_tokenizer()

    print("Загрузка модели...")

    model = load_model()

    print("Создание SFTTrainer...")

    trainer = create_sft_trainer(model, tokenizer, dataset)

    print("Старт обучения...")

    trainer.train()

    print("Сохранение адаптера...")

    trainer.save_model(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)

    print("Готово")


if __name__ == "__main__":
    main()