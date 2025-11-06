"""
Модуль с PyTorch Dataset для задачи предсказания следующего токена.
"""

import logging
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class NextTokenDataset(Dataset):
    def __init__(
        self,
        texts: List[List[str]],
        tokenizer,
        min_length: int = 2
    ):
        self.tokenizer = tokenizer
        self.min_length = min_length

        self.texts = [t for t in texts if len(t) >= min_length]

        skipped = len(texts) - len(self.texts)
        if skipped > 0:
            logger.warning(
                f"Пропущено {skipped} последовательностей короче {min_length} токенов"
            )

        logger.info(f"Dataset создан с {len(self.texts)} примерами")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.texts[idx]

        try:
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(target_ids, dtype=torch.long)
            }
        except Exception as e:
            logger.error(f"Ошибка обработки примера {idx}: {e}")
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long)
            }

def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    def get_length_group(item: Dict[str, torch.Tensor]) -> int:
        length = len(item["input_ids"])
        if length <= 10:
            return 0
        elif length <= 20:
            return 1
        elif length <= 40:
            return 2
        else:
            return 3
    
    batch.sort(key=lambda x: (get_length_group(x), len(x["input_ids"])), reverse=True)
    
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded
    }