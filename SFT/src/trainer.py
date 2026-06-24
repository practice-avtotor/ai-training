from trl import SFTTrainer, SFTConfig
from ...config import (
    SFT_OUTPUT_DIR, SFT_LEARNING_RATE, SFT_EPOCHS,
    BATCH_SIZE, GRAD_ACCUMULATION, SFT_MAX_LENGTH
)


def format_example(example, tokenizer):
    """
    Преобразует messages в текст по шаблону Qwen (Chat Template)
    """

    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )


def create_sft_trainer(model, tokenizer, dataset):
    # Упаковываем в формат для Qwen
    dataset = dataset.map(
        lambda x: {
            "text": format_example(x, tokenizer)
        }
    )

    # Настройки параметров обучения
    args = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=SFT_LEARNING_RATE,
        num_train_epochs=SFT_EPOCHS,
        bf16=True,
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        optim="paged_adamw_8bit",
        report_to="none",
        max_length=SFT_MAX_LENGTH
    )

    # Возвращаем оркестратор (тренера)
    return SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        processing_class=tokenizer,
    )