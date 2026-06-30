from pathlib import Path

#
# Пути
#

BASE_MODEL = "llm_qwen3_8B"
FINETUNED_MODEL = "model/qwen3_8B_final"
TEST_DATASET = "data/test_data.jsonl"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

#
# Генерация
#

MAX_NEW_TOKENS = 1024       # Максимальное количество токенов, должно хватить для тестовых примеров

# Для строгой оценки отключаем семплирование
TEMPERATURE = None
TOP_P = None
DO_SAMPLE = False

REPETITION_PENALTY = 1.0    # Отключаем штраф за повторения
BATCH_SIZE = 8

#
# Запуск модели
#

USE_4BIT = True             # Используем квантование
USE_BF16 = False