# Дообучение LLM для перевода 

> **Статус обучения модели:** Подготовка к обучению

> **Статус готовности данных:** Сбор данных 

## Стек

- Python 3.13
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT (QLoRA)
- TRL

## Загрузка необходимых библиотек

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121 \
  && pip install -r requirements.txt