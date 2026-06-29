from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from config import (
    SFT_OUTPUT_DIR, SFT_LEARNING_RATE, SFT_EPOCHS,
    BATCH_SIZE, GRAD_ACCUMULATION, SFT_MAX_LENGTH
)


def create_sft_trainer(model, tokenizer, dataset):
    """
    Стабильный Trainer с правильной передачей labels для расчета Loss
    """

    def tokenize_function(example):
        # Формируем финальный текст через оригинальный Chat Template
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

        # Токенизируем текст
        tokenized = tokenizer(
            text,
            max_length=SFT_MAX_LENGTH,
            truncation=True,
            padding=False
        )

        # Явно создаем labels внутри словаря до коллатора
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    print("Токенизация SFT датасета...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=False,
        remove_columns=dataset.column_names
    )

    # Настраиваем коллатор, указываем явный id для игнорирования паддингов
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    args = TrainingArguments(
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
        remove_unused_columns=False
    )

    return Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=args,
        data_collator=data_collator
    )