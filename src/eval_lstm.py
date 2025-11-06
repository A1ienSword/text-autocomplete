"""
Модуль для оценки качества LSTM модели.
"""
import logging
import torch
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import evaluate
import numpy as np

logger = logging.getLogger(__name__)

def evaluate_rouge(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    pad_token_id: int = 0,
    max_samples: int = 2000,
    cutoff_ratio: float = 0.75,
    temperature: float = 1.0,
    top_k: int = 20,
) -> Dict[str, float]:
    model.eval()
    preds, refs = [], []
    processed = 0

    logger.info(f"Оценка модели на {max_samples} примерах...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating ROUGE"):
            if processed >= max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = (input_ids != pad_token_id).sum(dim=1).tolist()

            for i, L in enumerate(lengths):
                if processed >= max_samples:
                    break
                if L < 4:
                    continue
                cutoff = max(1, int(L * cutoff_ratio))
                context = input_ids[i:i+1, :cutoff]
                target_len = L - cutoff

                if target_len <= 0:
                    continue

                try:
                    generated_ids = model.generate_tokens(
                        context,
                        max_new_tokens=target_len,
                        temperature=temperature,
                        top_k=top_k
                    )

                    new_tokens = generated_ids[0, cutoff:].tolist()
                    ref_tokens = labels[i, cutoff:L].tolist()
                    new_tokens = [t for t in new_tokens if t != pad_token_id]
                    ref_tokens = [t for t in ref_tokens if t != pad_token_id]
                    ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                    pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    preds.append(pred_text)
                    refs.append(ref_text)
                    processed += 1

                except Exception as e:
                    logger.warning(f"Ошибка при генерации для примера {processed}: {e}")
                    continue

    # Вычисление ROUGE
    if len(preds) == 0:
        logger.warning("Нет успешно обработанных примеров!")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    try:
        ROUGE = evaluate.load("rouge")
        rouge_scores = ROUGE.compute(predictions=preds, references=refs)

        logger.info(f"\nРезультаты ROUGE:")
        logger.info(f"  ROUGE-1: {rouge_scores['rouge1']}")
        logger.info(f"  ROUGE-2: {rouge_scores['rouge2']}")
        logger.info(f"  ROUGE-L: {rouge_scores['rougeL']}")

        return rouge_scores

    except Exception as e:
        logger.error(f"Ошибка вычисления ROUGE: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
