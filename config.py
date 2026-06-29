MODEL_NAME = "llm_qwen3_8B" # папка с исходной моделью
DATA_DIR = "data"
# OUTPUT_DIR = "output/qwen_cpt" # если есть CPT, используем новую модель
OUTPUT_DIR = "llm_qwen3_8B" # если нет CPT, используем исходную модель

MAX_LENGTH = 512       # максимальная длина в токенах

BLOCK_SIZE = 1024       # размер контекста в токенах

LORA_R = 16             # ранг матрицы
LORA_ALPHA = 32         # коэффициент масштабировнаия
LORA_DROPOUT = 0.05     # вероятность дропаута (защита от переобучения)

BATCH_SIZE = 1          # размер микро батча

GRAD_ACCUMULATION = 16  # шаг для накопления градиентов

LEARNING_RATE = 2e-5    # шаг градиентного спуска

EPOCHS = 3              # количество эпох


#
# Параметры для SFT
#

SFT_DATASET = "data/train.jsonl"
SFT_OUTPUT_DIR = "output/qwen_sft"

SFT_MAX_LENGTH = 512       # максимальная длина в токенах

SFT_LEARNING_RATE = 1e-5    # шаг градиентного спуска

SFT_EPOCHS = 2              # колчество эпох