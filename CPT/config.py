MODEL_NAME = "Qwen/Qwen3-14B"
DATA_DIR = "data"
OUTPUT_DIR = "output/qwen_cpt"

MAX_LENGTH = 2048       # максимальная длина в токенах

BLOCK_SIZE = 2048       # размер контекста в токенах

LORA_R = 32             # ранг матрицы
LORA_ALPHA = 64         # коэффициент масштабировнаия
LORA_DROPOUT = 0.05     # вероятность дропаута (защита от переобучения)

BATCH_SIZE = 1          # размер микро батча

GRAD_ACCUMULATION = 16  # шаг для накопления градиентов

LEARNING_RATE = 2e-5    # шаг градиентного спуска

EPOCHS = 3              # количество эпох