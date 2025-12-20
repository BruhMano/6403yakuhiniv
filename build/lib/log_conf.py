"""
Конфигурация логирования для приложения обработки изображений собак.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
import os


def setup_logging(log_file: str = "dog_app.log"):
    """
    Настраивает систему логирования.
    
    Args:
        log_file: Путь к файлу логов
        console_level: Уровень логирования для консоли
    """
    # Создаем директорию для логов, если она не существует
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Создаем форматтеры
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Создаем логгер
    logger = logging.getLogger('DogImageProcessor')
    logger.setLevel(logging.DEBUG)  # Самый низкий уровень для корневого логгера
    
    # Удаляем существующие обработчики
    logger.handlers.clear()
    
    # Файловый обработчик (DEBUG уровень, подробные логи)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Консольный обработчик (INFO уровень, краткие логи)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger