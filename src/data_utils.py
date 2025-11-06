"""
Модуль для загрузки и предобработки данных датасета sentiment140.

Этот модуль содержит функции для:
- Загрузки конфигурации из YAML
- Загрузки сырых данных
- Предобработки текста (очистка, нормализация)
- Разбиения данных на train/val/test
- Сохранения обработанных данных
"""

import json
import logging
import re
import yaml
from pathlib import Path
from typing import List, Tuple, Union, Any, Optional
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_config(config_path: Path) -> Any:
    if not config_path.is_file():
        logger.error(f"Файл конфигурации не найден: {config_path}")
        raise FileNotFoundError(f"Не найден файл по пути: {config_path}")

    try:
        with config_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Конфигурация успешно загружена из {config_path}")
        return config
    except Exception as e:
        logger.error(f"Ошибка при обработки конфигурации: {e}")
        raise

def load_data(data_path: Path) -> List[str]:
    if not data_path.is_file():
        logger.error(f"Файл данных не найден: {data_path}")
        raise FileNotFoundError(f"Не найден файл по пути: {data_path}")

    try:
        with data_path.open('r', encoding='utf-8') as f:
            data = f.readlines()
        logger.info(f"Загружено {len(data)} строк из {data_path}")
        return data
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        raise

def preprocess_text_simple(text: str) -> str:
    if not text or not isinstance(text, str):
        logger.warning(f"Получен невалидный текст: {type(text)}")
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    text = re.sub(r'[^a-z0-9\s\.,!?\;:\'\"<>_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_text_with_tokenizer(text: str, tokenizer: AutoTokenizer) -> List[str]:
    cleaned_text = preprocess_text_simple(text)

    if not cleaned_text:
        return []

    try:
        tokens = tokenizer.tokenize(cleaned_text)
        return tokens
    except Exception as e:
        logger.warning(f"Ошибка токенизации: {e}")
        return []


def process_data(data: List[str], tokenizer: Optional[AutoTokenizer] = None, min_length: int = 3) -> List[Union[str, List[str]]]:
    processed_data = []

    for sentence in tqdm(data, desc=f"Обработка предложений c минимальной длинной {min_length}"):
        if tokenizer:
            processed = preprocess_text_with_tokenizer(sentence, tokenizer)
        else:
            processed = preprocess_text_simple(sentence)
        if isinstance(processed, list):
            if len(processed) >= min_length:
                processed_data.append(processed)
        elif isinstance(processed, str):
            if len(processed.split()) >= min_length:
                processed_data.append(processed)

    logger.info(f"Обработано: {len(processed_data)}")
    return processed_data

def save_processed_data(data: List, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            logger.info(f"Удаление существующего файла: {path}")
            path.unlink()
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Данные сохранены в {path} ({len(data)} записей)")
    except Exception as e:
        logger.error(f"Ошибка сохранения данных: {e}")
        raise
        
def process_dataset(config_path: str = './configs/config.yaml', tokenizer: Optional[AutoTokenizer] = None, force_reprocess: bool = False) -> List[Union[str, List[str]]]:
    config = get_config(Path(config_path))
    processed_path = Path(config['data']['path']['processed'])

    if not force_reprocess and processed_path.exists():
        logger.info(f"Загрузка уже обработанных данных по пути: {processed_path}")
        try:
            with processed_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Ошибка загрузки: {e}")

    data = load_data(Path(config['data']['path']['raw']))
    processed_data = process_data(data, tokenizer)
    save_processed_data(processed_data, processed_path)

    return processed_data


def split_data(processed_data: List[Union[str, List[str]]], config_path: str = './configs/config.yaml') -> Tuple[List[Union[str, List[str]]], List[Union[str, List[str]]], List[Union[str, List[str]]]]:
    config = get_config(Path(config_path))

    logger.info(f"Разбиение {len(processed_data)} примеров на train/val/test")

    train_texts, temp_texts = train_test_split(
        processed_data,
        test_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )

    val_texts, test_texts = train_test_split(
        temp_texts,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    save_processed_data(train_texts, Path(config['data']['path']['train']))
    save_processed_data(val_texts, Path(config['data']['path']['val']))
    save_processed_data(test_texts, Path(config['data']['path']['test']))

    return train_texts, val_texts, test_texts
