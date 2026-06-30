import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import remove_thinking
from config_eval import *


class InferenceEngine:
    """
    Класс загрузки модели и токенизатора
    """
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Загрузка токенизатора: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Настройка паддинга для батчевой генерации
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Используем квантование
        if USE_4BIT:
            print("Загрузка модели в 4-bit...")
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant,
                device_map="auto",
                trust_remote_code=True
            )

        else:
            dtype = torch.bfloat16 if USE_BF16 else torch.float16
            print("Загрузка модели с полной точностью...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )

        # Перевод модели в режим инференса
        self.model.eval()

    def generate_batch(self, batch_messages):
        """
        Генерация для батча
        """

        # Преобразование списка сообщений JSON в строку с разделителями
        prompts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            ) for msgs in batch_messages
        ]

        # Превразаем текстовые строки в числовые ID токенов
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",    # результаты в виде тензоров PyTorch
            # включаем выравнивание и обрезку, если текст превысил контекст модели
            padding=True,
            truncation=True
        ).to(self.model.device)

        # Находим ID токена <think>
        bad_words_ids = self.tokenizer(
            ["<think>"],
            add_special_tokens=False
        ).input_ids

        # Отключаем расчет градиентов
        with torch.no_grad():
            # запускаем генерацию
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                bad_words_ids=bad_words_ids
            )

        # Отрезаем промпт и декодируем
        answers = []
        for i, output in enumerate(outputs):
            input_len = inputs.input_ids[i].shape[0]
            generated = output[input_len:]
            answer = self.tokenizer.decode(generated, skip_special_tokens=True)
            answers.append(remove_thinking(answer))

        return answers