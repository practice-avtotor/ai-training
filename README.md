# Дообучение LLM для перевода 

> **Статус обучения модели:** SFT-обучение

> **Статус готовности данных:** Сбор данных завершен 

## Стек

- Python 3.13
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT (QLoRA)
- TRL

## Используемое оборудование

- RTX 3080 (10 GB)

## Загрузка необходимых библиотек

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
  && pip install -r requirements.txt