"""
Модуль для обучения LSTM модели.
"""
import logging
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    pad_token_id: int = 0,
    save_dir: str = "./models",
    patience: int = 5,
) -> nn.Module:
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    best_model_path = save_dir / "best_model.pt"
    train_losses = []
    val_losses = []
    learning_rates = []

    logger.info(f"Начало обучения на {epochs} эпох")

    for epoch in range(epochs):
        # === Обучение ===
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # === Валидация ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        scheduler.step(avg_val_loss)
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, best_model_path)
            logger.info(f"Сохранена лучшая модель (val_loss: {avg_val_loss:.4f})")
        epoch_path = save_dir / f"model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), epoch_path)
        if early_stopping(-avg_val_loss): 
            logger.info(f"Early stopping на эпохе {epoch+1}")
            break
        torch.cuda.empty_cache()
    plt.figure(figsize=(15, 5))
    epochs_range = range(1, len(train_losses) + 1)

    # График 1: Losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training vs Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # График 2: Learning Rate
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, learning_rates, 'g-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # График 3: Overfitting Gap
    plt.subplot(1, 3, 3)
    gap = [val - train for train, val in zip(train_losses, val_losses)]
    plt.plot(epochs_range, gap, 'm-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Val Loss - Train Loss', fontsize=12)
    plt.title('Overfitting Gap', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Сохранение графиков
    plot_path = save_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"График сохранен в {plot_path}")
    plt.show()
    
    logger.info(f"Обучение завершено. Лучшая модель: {best_model_path}")
    return model
