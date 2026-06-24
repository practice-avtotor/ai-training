from transformers import Trainer, TrainingArguments
from ...config import (
    OUTPUT_DIR, BATCH_SIZE, GRAD_ACCUMULATION,
    LEARNING_RATE, EPOCHS
)


def create_trainer(model,dataset):

    # Настройки параметров обучения
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    # Возвращаем оркстратор
    return Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )