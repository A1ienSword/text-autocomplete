"""
Модуль для оценки качества трансформерной модели (DistilGPT2).
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def print_evaluation_results(
    model_name: str,
    scores: Dict[str, float]
) -> None:
    logger.info(f"\n=== Результаты оценки: {model_name} ===")
    logger.info(f"ROUGE-1: {scores['rouge1']:.4f}")
    logger.info(f"ROUGE-2: {scores['rouge2']:.4f}")
    logger.info(f"ROUGE-L: {scores['rougeL']:.4f}")


def compare_models(
    lstm_scores: Dict[str, float],
    transformer_scores: Dict[str, float]
) -> Dict[str, Any]:
    logger.info("СРАВНЕНИЕ МОДЕЛЕЙ")
    logger.info("="*60)
    logger.info(f"{'Модель':<20} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12}")
    logger.info("-" * 60)
    logger.info(
        f"{'LSTM':<20} {lstm_scores['rouge1']:<12.4f} "
        f"{lstm_scores['rouge2']:<12.4f} {lstm_scores['rougeL']:<12.4f}"
    )
    logger.info(
        f"{'DistilGPT2':<20} {transformer_scores['rouge1']:<12.4f} "
        f"{transformer_scores['rouge2']:<12.4f} {transformer_scores['rougeL']:<12.4f}"
    )
    logger.info("-" * 60)
    best_model = "DistilGPT2" if transformer_scores['rougeL'] > lstm_scores['rougeL'] else "LSTM"
    improvement = abs(transformer_scores['rougeL'] - lstm_scores['rougeL'])
    improvement_pct = improvement / max(lstm_scores['rougeL'], 0.0001) * 100
    logger.info(f"\nЛучшая модель: {best_model}")
    logger.info(f"Улучшение ROUGE-L: {improvement:.4f} ({improvement_pct:.2f}%)")
    return {
        "best_model": best_model,
        "improvement": improvement,
        "improvement_percent": improvement_pct,
        "lstm_scores": lstm_scores,
        "transformer_scores": transformer_scores
    }
