import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from config import MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT

# Предполагается использовать видеокарту RTX 3080ti 12GB

def load_base_quantized_model():
    """
    Загружает только чистую базовую модель в 4-бит
    """
    # Сжатие модели в 4 бита
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Загрузка модели
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    # Подготавливаем модель для kbit обучения
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def load_model():
    """
    Для CPT создает модель с пустым LoRA-адаптером
    """
    base_model = load_base_quantized_model()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def load_cpt_model(path):
    """
    Для SFT загружает чистую модель и готовый LoRA адаптер после CPT
    """
    base_model = load_base_quantized_model()

    # Подгружаем веса LoRa адаптера
    model = PeftModel.from_pretrained(base_model, path, is_trainable=True)

    model.print_trainable_parameters()
    return model