import os
import time
import asyncio
import aiohttp
import aiofiles
import cv2
from typing import List, Tuple, Optional
from PIL import Image
from io import BytesIO
import numpy as np
import concurrent.futures
from .dog_image import ColorDogImage, GrayscaleDogImage
from .image_processing import ImageProcessing
from .cv2_image_processing import Cv2ImageProcessing
import multiprocessing
from functools import wraps
from logging import getLogger

logger = getLogger('DogImageProcessor')

def async_time_logger(func):
    """
    Декоратор для измерения времени выполнения асинхронного метода.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        logger.debug(f"Начало выполнения {func.__name__}")
        result = await func(*args, **kwargs)
        end = time.time()
        logger.debug(f"Завершено {func.__name__} за {end - start:.2f} сек.")
        return result
    return wrapper


class DogImageProcessor:
    """
    Класс для асинхронной работы с Dog API, загрузки, обработки и сохранения изображений собак.
    """

    API_URL = "https://api.thedogapi.com/v1/images/search"

    def __init__(self, api_key: str, output_dir: str, grey: bool, limit: int, 
                 max_concurrent_downloads: int = 3):
        """
        Инициализирует процессор изображений собак.
        
        Args:
            api_key (str): Ключ для доступа к Dog API.
            output_dir (str): Директория для сохранения изображений.
            grey (bool): Если True — использовать GrayscaleDogImage, иначе ColorDogImage.
            limit (int): Количество изображений для загрузки.
            max_concurrent_downloads (int): Максимальное количество одновременных загрузок.
        """
        self._api_key = api_key
        self._output_dir = output_dir
        self._grey = grey
        self._limit = limit
        self._max_concurrent_downloads = max_concurrent_downloads
        self._image_data: List[Tuple[int, str, str]] = []  # (index, breed, url)
        os.makedirs(self._output_dir, exist_ok=True)

    @property
    def limit(self) -> int:
        """int: Возвращает лимит загружаемых изображений."""
        return self._limit

    @async_time_logger
    async def _fetch_image_urls(self) -> None:
        """
        Асинхронно загружает URL изображений собак из API.
        
        Примечание:
            Порядковые номера изображений определяются здесь и сохраняются.
        """
        headers = {"x-api-key": self._api_key}
        params = {"limit": self.limit, "has_breeds": 1}

        logger.debug("Начало получения URL изображений из API")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.API_URL, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Определяем порядковые номера в самом начале
                for idx, item in enumerate(data):
                    url = item["url"]
                    breed = item["breeds"][0]["name"].lower().replace(" ", "_") if item["breeds"] else "unknown"
                    self._image_data.append((idx, breed, url))
                    
        logger.info(f"Получено {len(self._image_data)} URL изображений")

    async def _download_single_image(self, session: aiohttp.ClientSession, 
                                idx: int, breed: str, url: str) -> Optional[Tuple[int, np.ndarray, str]]:
        """
        Загружает одно изображение асинхронно.
        """

        start_time = time.time()
        logger.debug(f"Загрузка изображения №{idx} началась")
        
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                # Читаем данные асинхронно
                img_data = await response.read()
                
                # Используем run_in_executor для блокирующей операции
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(
                    None,  # Используем стандартный ThreadPoolExecutor
                    self._process_image_data,
                    img_data
                )
                image_array = np.array(image)
                
                end_time = time.time()
                logger.debug(f"Загрузка изображения №{idx} завершилась за {end_time - start_time:.2f} сек. ")
                
                return (idx, image_array, breed)
            else:
                logger.error(f"Изображение №{idx} не было загружено, статус: {response.status}")
                return None

    def _process_image_data(self, img_data: bytes) -> Image.Image:
        """Синхронный метод для обработки изображений в executor."""
        return Image.open(BytesIO(img_data)).convert("RGB")

    @async_time_logger
    async def _download_images(self) -> List[Tuple[int, np.ndarray, str]]:
        """
        Асинхронно загружает все изображения.
        """
        if not self._image_data:
            await self._fetch_image_urls()
            
        downloaded_images = []

        logger.debug(f"Начало загрузки {len(self._image_data)} изображений")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for idx, breed, url in self._image_data:
                task = asyncio.create_task(
                    self._download_single_image(session, idx, breed, url)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Ошибка во время загрузки: {result}")
                elif result is not None:
                    downloaded_images.append(result)
        
        # Сортируем по индексу для сохранения порядка
        downloaded_images.sort(key=lambda x: x[0])
        logger.info(f"Успешно загружено {len(downloaded_images)} изображений")

        return downloaded_images

    def _process_single_image_sync(self, idx: int, img_array: np.ndarray, 
                                  breed: str) -> List[Tuple[str, np.ndarray]]:
        """
        Синхронная обработка одного изображения.
        
        Args:
            idx: Порядковый номер изображения.
            img_array: Массив изображения.
            breed: Порода собаки.
            processor_pid: PID процесса обработки.
            
        Returns:
            Список кортежей (суффикс, изображение).
        """
        results = []
        process_pid = os.getpid()

        # Создаем объект DogImage
        start_time = time.time()
        logger.debug(f"Начало обработки изображения №{idx} в процессе с PID={process_pid}")
        
        dog = (
            ColorDogImage(img_array, breed, "", ImageProcessing())
            if not self._grey
            else GrayscaleDogImage(img_array, breed, "", ImageProcessing())
        )
        
        # Сохраняем оригинал
        results.append(("original", dog.image.copy()))
        
        # Модифицируем своими методами
        logger.debug(f"Вычисление границ кастомными методами для изображения №{idx} началось (PID {process_pid})")
        edges = GrayscaleDogImage(dog.edges(), dog.breed, dog.url, dog.processor)
        new_image = dog + edges
        results.append(("with_edges", new_image.image.copy()))
        
        # Модифицируем методом cv2
        logger.debug(f"Вычисление границ методами OpenCV для изображения №{idx} началось (PID {process_pid})")
        dog.processor = Cv2ImageProcessing()
        edges.image = dog.edges()
        new_image = dog + edges
        results.append(("with_edges_cv2", new_image.image.copy()))
        
        end_time = time.time()
        logger.debug(f"Обработка изображения №{idx} завершена за {end_time - start_time:.2f} сек. (PID {process_pid})")
        
        return [(f"{idx}_{breed}_{suffix}.png", img) for suffix, img in results]

    @async_time_logger
    async def _process_images(self, downloaded_images: List[Tuple[int, np.ndarray, str]]) -> List:
        """
        Асинхронная обработка изображений.
        """
        if not downloaded_images:
            logger.warning("Нет изображений для обработки")
            return []
        
        all_results = []

        # Используем ProcessPoolExecutor для CPU-bound операций
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(downloaded_images))) as executor:
            loop = asyncio.get_event_loop()
            
            logger.debug(f"Начало обработки {len(downloaded_images)} изображений в параллельных процессах")
            logger.debug(f"PID основного процесса: {os.getpid()}")
            logger.debug(f"Количество яред CPU: {multiprocessing.cpu_count()}")
            
            # Подготавливаем задачи для асинхронного выполнения
            tasks = []
            for idx, img_array, breed in downloaded_images:
                # Запускаем обработку в отдельном процессе
                task = loop.run_in_executor(
                    executor, 
                    self._process_single_image_sync, 
                    idx, img_array, breed
                )
                tasks.append(task)
            
            # Ждем завершения всех задач
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Обрабатываем результаты
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Ошибка при обработке изображения №{idx}: {str(result)}")
                else:
                    logger.debug(f"Обработка изображения №{idx} завершена")
                    all_results.extend(result)
            logger.info(f"Успешно обработано {len(downloaded_images)} изображений")
            return all_results

    async def _save_single_image(self, filename: str, img_array: np.ndarray) -> None:
        """
        Асинхронно сохраняет одно изображение.
        
        Args:
            filename: Имя файла.
            img_array: Массив изображения.
        """
        filepath = os.path.join(self._output_dir, filename)
        
        # Конвертируем RGB в BGR для OpenCV
        bgr_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Кодируем изображение в PNG в памяти
        success, encoded_image = cv2.imencode('.png', bgr_image)
        
        if success:
            # Асинхронно записываем закодированные данные
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(encoded_image.tobytes())
            logger.debug(f"Изображение сохранено в файле {filename}")
        else:
            logger.error(f"Ошибка при кодировании изображения {filename}")

    @async_time_logger
    async def _save_all_images(self, images_to_save: List[Tuple[str, np.ndarray]]) -> None:
        """
        Асинхронно сохраняет все изображения.
        """
        if not images_to_save:
            logger.warning("Нет изображений для сохранения")
            return
        
        logger.debug(f"Начало сохранения {len(images_to_save)} изображений")
        # Создаем задачи для асинхронного сохранения
        tasks = []
        for filename, img_array in images_to_save:
            task = asyncio.create_task(self._save_single_image(filename, img_array))
            tasks.append(task)
        
        # Ждем завершения всех задач сохранения
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Успешно сохранено {len(images_to_save)} изображений")

    @async_time_logger
    async def process_and_save(self) -> None:
        """
        Основной метод для асинхронной обработки и сохранения изображений.
        """
        total_start_time = time.time()
        
        logger.info(f"Запуск обработки {self._limit} изображений")
        logger.info(f"Выходная директория: {self._output_dir}")
        
        try:
            # 1. Получаем URL изображений
            logger.info("Шаг 1: Получение URL изображений...")
            await self._fetch_image_urls()
            
            # 2. Асинхронно загружаем изображения
            logger.info(f"Шаг 2: Загрузка {len(self._image_data)} изображений...")
            downloaded_images = await self._download_images()
            
            if not downloaded_images:
                logger.warning("Нет загруженных изображений, завершение...")
                return
            
            # 3. Параллельно обрабатываем изображения
            logger.info(f"Шаг 3: Обработка {len(downloaded_images)} изображений...")
            images_to_save = await self._process_images(downloaded_images)
            
            if not images_to_save:
                logger.warning("Нет обработанных изображений, завершение...")
                return
            
            # 4. Асинхронно сохраняем все изображения
            logger.info(f"Шаг 4: Сохранение {len(images_to_save)} изображений...")
            await self._save_all_images(images_to_save)
            
        except Exception as e:
            logger.critical(f"Критическая ошибка при обработке: {str(e)}", exc_info=True)
        finally:
            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            
            logger.info("Обработка завершена!")
            logger.info(f"Общее время: {total_time:.2f} секунд")