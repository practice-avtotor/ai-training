import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from ...config import MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT

# Предполагается использовать видеокарту RTX 3080ti 12GB

def load_model():
    """
    Загрузка модели в память видеокарты
    """

    # Сжатие модели в 4 бита
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Настройка конфига LoRa
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    # Загрузка модели
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Подготавливаем модель для kbit обучения
    model = prepare_model_for_kbit_training(model)

    # оборачиваем модель в PEFT
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model