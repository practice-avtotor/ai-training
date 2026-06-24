from config import DATA_DIR, OUTPUT_DIR
from src.dataset import build_dataset
from src.tokenizer import tokenize_dataset, load_tokenizer
from src.model import load_model
from src.cpt_trainer import create_trainer

"""
Запуск цикла обучения CPT
"""

def main():
    print("Загрузка данных...")

    dataset = build_dataset(DATA_DIR)

    print(f"Документов: {len(dataset)}")

    print("Токенизация...")

    lm_dataset = tokenize_dataset(dataset)

    print("Загрузка модели...")

    model = load_model()

    print("Создание Trainer...")

    trainer = create_trainer(model, lm_dataset)

    print("Старт обучения...")

    trainer.train()

    print("Сохранение адаптера...")

    # Сохраняем модель и токенизатор
    trainer.save_model(OUTPUT_DIR)
    tokenizer = load_tokenizer()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Готово")


if __name__ == "__main__":
    main()