"""
Модуль с LSTM моделью для генерации текста.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        pad_token_id: int = 0,
        dropout: float = 0.2,
        embed_dropout: float = 0.1,
        bidirectional: bool = False,
        use_layer_norm: bool = False
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_token_id
        )
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )
        if use_layer_norm:
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.layer_norm = nn.LayerNorm(lstm_output_dim)
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, vocab_size)
        logger.info(
            f"LSTMModel создана: vocab={vocab_size}, embed={embed_dim}, "
            f"hidden={hidden_dim}, layers={num_layers}, bidirectional={bidirectional}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(input_ids)
        emb = self.embed_dropout(emb)
        output, hidden = self.lstm(emb, hidden)
        if self.use_layer_norm:
            output = self.layer_norm(output)
        logits = self.fc(output)
        return logits, hidden

    def _generate_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        if temperature <= 0:
            temperature = 1.0
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            vals, idxs = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(vals, dim=-1)
            next_token = idxs.gather(-1, torch.multinomial(probs, 1))
        elif top_p is not None and 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            sorted_logits[sorted_indices_to_remove] = float('-inf')
            probs = F.softmax(sorted_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = sorted_indices.gather(-1, next_token_idx)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        return next_token


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> str:
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        generated_ids = self.generate_tokens(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        generated_text = tokenizer.decode(
            generated_ids[0].tolist(),
            skip_special_tokens=True
        )

        return generated_text


    @torch.no_grad()
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        logits, hidden = self.forward(input_ids)
        last_token = input_ids[:, -1:]
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits, hidden = self.forward(last_token, hidden)
            next_token_logits = logits[:, -1, :]
            next_token = self._generate_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            generated = torch.cat([generated, next_token], dim=1)
            last_token = next_token

        return generated
