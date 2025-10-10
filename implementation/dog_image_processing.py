import os
import time
import requests
import cv2
from typing import List, Tuple
from PIL import Image
from io import BytesIO
import numpy as np
from implementation.dog_image import DogImage, ColorDogImage, GrayscaleDogImage
from implementation import ImageProcessing, Cv2ImageProcessing


def time_logger(func):
    """
    Декоратор для измерения времени выполнения метода и логирования.
    
    Args:
        func (callable): Декорируемая функция.
        
    Returns:
        callable: Обёрнутая функция с логированием времени выполнения.
        
    Examples:
        >>> @time_logger
        ... def my_function():
        ...     time.sleep(1)
        [LOG] Начало выполнения my_function
        [LOG] Завершено my_function за 1.00 сек.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"[LOG] Начало выполнения {func.__name__}")
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[LOG] Завершено {func.__name__} за {end - start:.2f} сек.")
        return result
    return wrapper


class DogImageProcessor:
    """
    Класс для работы с Dog API, загрузки, обработки и сохранения изображений собак.
    
    Инкапсулирует:
        - взаимодействие с API,
        - управление обработкой изображений,
        - сохранение результатов в файлы.
        
    Attributes:
        API_URL (str): Базовый URL для доступа к Dog API.
        _api_key (str): Ключ аутентификации для API.
        _output_dir (str): Директория для сохранения обработанных изображений.
        _grey (bool): Флаг использования grayscale-режима.
        _limit (int): Максимальное количество загружаемых изображений.
        _processor: Процессор для обработки изображений (опционально).
        
    Raises:
        requests.HTTPError: При ошибках HTTP-запросов к API.
        OSError: При проблемах создания директории или сохранения файлов.
    """

    API_URL = "https://api.thedogapi.com/v1/images/search"

    def __init__(self, api_key: str, output_dir: str, grey: bool, limit: int, operation: str):
        """
        Инициализирует процессор изображений собак.
        
        Args:
            api_key (str): Ключ для доступа к Dog API.
            output_dir (str): Директория для сохранения изображений.
            grey (bool): Если True — использовать GrayscaleDogImage, иначе ColorDogImage.
            limit (int): Количество изображений для загрузки.
            
        Note:
            Автоматически создает выходную директорию, если она не существует.
        """
        self._api_key = api_key
        self._output_dir = output_dir
        self._grey = grey
        self._limit = limit
        self._operation = operation
        os.makedirs(self._output_dir, exist_ok=True)

    @time_logger
    def _fetch_images(self) -> List[Tuple[np.ndarray, str, str]]:
        """
        Загружает изображения собак из API.
        
        Process:
            1. Выполняет HTTP-запрос к Dog API с аутентификацией
            2. Обрабатывает JSON-ответ
            3. Загружает каждое изображение по URL
            4. Конвертирует в RGB формат и преобразует в numpy array
            
        Returns:
            List[Tuple[np.ndarray, str, str]]: Список кортежей (изображение, порода, URL).
            
        Raises:
            requests.HTTPError: При ошибках HTTP-запросов.
            ValueError: При проблемах парсинга JSON или обработки изображений.
        """
        headers = {"x-api-key": self._api_key}
        params = {"limit": self._limit, "has_breeds": 1}
        response = requests.get(self.API_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data:
            url = item["url"]
            breed = item["breeds"][0]["name"].lower().replace(" ", "_") if item["breeds"] else "unknown"
            img_resp = requests.get(url)
            image = Image.open(BytesIO(img_resp.content)).convert("RGB")
            img_resp.raise_for_status()
            image_array = np.array(image)
            results.append((image_array, breed, url))
        return results

    @time_logger
    def process_and_save(self) -> None:
        """
        Обрабатывает и сохраняет изображения собак.
        
        Process:
            1. Загружает изображения через API
            2. Для каждого изображения:
                - Создает объект DogImage соответствующего типа
                - Сохраняет оригинальное изображение
                - Применяет детекцию границ пользовательскими методами
                - Сохраняет результат
                - Применяет детекцию границ методами OpenCV
                - Сохраняет результат
                
        Output:
            Для каждого исходного изображения создается три файла:
            - {idx}_{breed}_original.png - исходное изображение
            - {idx}_{breed}_with_edges.png - с границами (пользовательская обработка)
            - {idx}_{breed}_with_edges_cv2.png - с границами (OpenCV обработка)
            
        Note:
            Время выполнения каждого этапа логируется через декоратор @time_logger.
        """
        raw_images = self._fetch_images()
        for idx, (img_array, breed, url) in enumerate(raw_images):
            # Создаём объект DogImage нужного типа
            dog: DogImage = (
                ColorDogImage(img_array, breed, url, ImageProcessing())
                if not self._grey
                else GrayscaleDogImage(img_array, breed, url, self._processor)
            )

            # Сохраняем исходное изображение 
            self._save_image(dog.image, idx, breed, "original")

            # Модифицируем своими методами и сохраняем
            if self._operation == 'edges':
                filter_image = dog.edges()
            elif self._operation == 'corners':
                filter_image = dog.corners()
            elif self._operation == 'circles':
                filter_image = dog.circles()
            else: filter_image = np.zeros(dog.image.shape)
            filter_dog = ColorDogImage(filter_image, dog.breed, dog.url, dog.processor)
            new_image = dog + filter_dog
            self._save_image(new_image.image, idx, breed, f"with_{self._operation}")

            dog.processor = Cv2ImageProcessing()

            # Модифицируем методом cv2 и сохраняем
            if self._operation == 'edges':
                filter_image = dog.edges()
            elif self._operation == 'corners':
                filter_image = dog.corners()
            elif self._operation == 'circles':
                filter_image = dog.circles()
            else: filter_image = np.zeros(dog.image.shape)
            filter_dog = ColorDogImage(filter_image, dog.breed, dog.url, dog.processor)
            new_image = dog + filter_dog
            self._save_image(new_image.image, idx, breed, f"with_{self._operation}_cv2")

    def _save_image(self, img_array: np.ndarray, idx: int, breed: str, suffix: str) -> None:
        """
        Сохраняет изображение в файл с заданным именем.
        
        Args:
            img_array (np.ndarray): Изображение для сохранения в формате RGB.
            idx (int): Порядковый номер изображения.
            breed (str): Название породы для включения в имя файла.
            suffix (str): Суффикс файла, описывающий тип обработки.
            
        Note:
            Автоматически конвертирует RGB в BGR для корректного сохранения через OpenCV.
            Использует формат PNG для сохранения без потерь качества.
            
        Raises:
            OSError: При проблемах записи файла.
            cv2.error: При проблемах обработки изображения OpenCV.
        """
        filename = f"{idx}_{breed}_{suffix}.png"
        filepath = os.path.join(self._output_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        print(f"[SAVE] {filepath}")