"""
Модуль для работы с предобученной моделью DistilGPT2.
"""
import logging
import torch
from typing import Dict, List, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import evaluate

logger = logging.getLogger(__name__)
class DistilGPT2:    
    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info("Загрузка предобученной модели distilgpt2...")
        try:
            self.generator = pipeline(
              "text-generation", 
              model="distilgpt2", 
              device=0 if device == "cuda" else -1
              )
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Модель загружена успешно на {self.device}!")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def generate(
        self,
        text: str,
        max_new_tokens: int = 20,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
        num_return_sequences: int = 1
    ) -> str:
        try:
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "num_return_sequences": num_return_sequences
            }

            if do_sample:
                generation_kwargs.update({
                    "top_k": top_k,
                    "top_p": top_p
                })

            result = self.generator(text, **generation_kwargs)

            return result[0]["generated_text"]

        except Exception as e:
            logger.error(f"Ошибка генерации текста: {e}")
            return text
    
    def evaluate_rouge(
        self,
        val_texts: List[str],
        max_samples: int = 2000,
        max_new_tokens: int = 20,
        cutoff_ratio: float = 0.75,
        min_tokens: int = 4,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
        show_examples: bool = True,
        num_examples: int = 5
    ) -> Dict[str, float]:
        preds, refs = [], []
        examples = []
        processed = 0

        logger.info(f"Оценка distilgpt2 на {min(len(val_texts), max_samples)} примерах...")

        for i, text in enumerate(tqdm(val_texts[:max_samples], desc="Evaluating")):
            if processed >= max_samples:
                break

            if isinstance(text, list):
                continue

            tokens = self.tokenizer.tokenize(text)

            if len(tokens) < min_tokens:
                continue

            cutoff = max(1, int(len(tokens) * cutoff_ratio))
            context_tokens = tokens[:cutoff]
            target_tokens = tokens[cutoff:]

            if len(target_tokens) == 0:
                continue

            context_text = self.tokenizer.convert_tokens_to_string(context_tokens)
            target_text = self.tokenizer.convert_tokens_to_string(target_tokens)

            try:
                generated_text = self.generate(
                    context_text,
                    max_new_tokens=len(target_tokens) + 5,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature
                )

                generated_continuation = generated_text[len(context_text):].strip()

                preds.append(generated_continuation)
                refs.append(target_text)
                processed += 1
                if show_examples and len(examples) < num_examples:
                    examples.append({
                        'context': context_text,
                        'generated': generated_continuation,
                        'target': target_text
                    })

            except Exception as e:
                logger.warning(f"Ошибка при генерации для примера {i}: {e}")
                continue

        logger.info(f"Обработано {len(preds)} примеров")

        if show_examples and examples:
            logger.info("\n=== Примеры генерации DistilGPT2 ===")
            for i, ex in enumerate(examples, 1):
                logger.info(f"\nПример {i}:")
                logger.info(f"  Контекст: {ex['context']}")
                logger.info(f"  Генерация: {ex['generated']}")
                logger.info(f"  Целевой: {ex['target']}")

        if len(preds) == 0:
            logger.warning("Нет успешно обработанных примеров!")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        try:
            ROUGE = evaluate.load("rouge")
            rouge_scores = ROUGE.compute(predictions=preds, references=refs)

            logger.info(f"\nРезультаты ROUGE для DistilGPT2:")
            logger.info(f"  ROUGE-1: {rouge_scores['rouge1']}")
            logger.info(f"  ROUGE-2: {rouge_scores['rouge2']}")
            logger.info(f"  ROUGE-L: {rouge_scores['rougeL']}")

            return rouge_scores

        except Exception as e:
            logger.error(f"Ошибка вычисления ROUGE: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
