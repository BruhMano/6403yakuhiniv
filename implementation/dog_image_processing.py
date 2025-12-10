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
from implementation.dog_image import DogImage, ColorDogImage, GrayscaleDogImage
from implementation import ImageProcessing, Cv2ImageProcessing
import multiprocessing
from functools import wraps
from datetime import datetime

def async_time_logger(func):
    """
    Декоратор для измерения времени выполнения асинхронного метода.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        print(f"[LOG] Начало выполнения {func.__name__}")
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"[LOG] Завершено {func.__name__} за {end - start:.2f} сек.")
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
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.API_URL, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Определяем порядковые номера в самом начале
                for idx, item in enumerate(data):
                    url = item["url"]
                    breed = item["breeds"][0]["name"].lower().replace(" ", "_") if item["breeds"] else "unknown"
                    self._image_data.append((idx, breed, url))
                    
        print(f"[INFO] Получено {len(self._image_data)} URL изображений")

    async def _download_single_image(self, session: aiohttp.ClientSession, 
                                idx: int, breed: str, url: str) -> Optional[Tuple[int, np.ndarray, str]]:
        """
        Загружает одно изображение асинхронно.
        """
        start_time = time.time()
        print(f"[DOWNLOAD] Downloading image {idx} started at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    # Читаем данные асинхронно
                    img_data = await response.read()
                    
                    # Используем run_in_executor для CPU-bound операции
                    loop = asyncio.get_event_loop()
                    image = await loop.run_in_executor(
                        None,  # Используем стандартный ThreadPoolExecutor
                        self._process_image_data,
                        img_data
                    )
                    image_array = np.array(image)
                    
                    end_time = time.time()
                    print(f"[DOWNLOAD] Downloading image {idx} finished in {end_time - start_time:.2f} сек. "
                        f"({datetime.now().strftime('%H:%M:%S.%f')[:-3]})")
                    
                    return (idx, image_array, breed)
                else:
                    print(f"[ERROR] Failed to download image {idx}, status: {response.status}")
                    return None
        except asyncio.TimeoutError:
            print(f"[ERROR] Timeout downloading image {idx}")
            return None
        except Exception as e:
            print(f"[ERROR] Error downloading image {idx}: {str(e)}")
            return None

    def _process_image_data(self, img_data: bytes) -> Image.Image:
        """Синхронный метод для обработки изображений в executor."""
        return Image.open(BytesIO(img_data)).convert("RGB")

    @async_time_logger
    async def _download_images(self) -> List[Tuple[int, np.ndarray, str]]:
        """
        Асинхронно загружает все изображения с ограничением на количество одновременных запросов.
        """
        if not self._image_data:
            await self._fetch_image_urls()
            
        downloaded_images = []
        
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
                    print(f"[ERROR] Exception during download: {result}")
                elif result is not None:
                    downloaded_images.append(result)
        
        # Сортируем по индексу для сохранения порядка
        downloaded_images.sort(key=lambda x: x[0])
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
        processor_pid = os.getpid()

        # Создаем объект DogImage
        start_time = time.time()
        print(f"[PROCESS] Processing for image {idx} started (PID {processor_pid}) "
              f"at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        dog: DogImage = (
            ColorDogImage(img_array, breed, "", ImageProcessing())
            if not self._grey
            else GrayscaleDogImage(img_array, breed, "", ImageProcessing())
        )
        
        # Сохраняем оригинал
        results.append(("original", dog.image.copy()))
        
        # Модифицируем своими методами
        print(f"[PROCESS] Convolution for image {idx} started (PID {processor_pid})")
        edges = GrayscaleDogImage(dog.edges(), dog.breed, dog.url, dog.processor)
        new_image = dog + edges
        results.append(("with_edges", new_image.image.copy()))
        
        # Модифицируем методом cv2
        print(f"[PROCESS] OpenCV processing for image {idx} started (PID {processor_pid})")
        dog.processor = Cv2ImageProcessing()
        edges.image = dog.edges()
        new_image = dog + edges
        results.append(("with_edges_cv2", new_image.image.copy()))
        
        end_time = time.time()
        print(f"[PROCESS] Processing for image {idx} finished in {end_time - start_time:.2f} сек. "
              f"(PID {processor_pid})")
        
        return [(f"{idx}_{breed}_{suffix}.png", img) for suffix, img in results]

    @async_time_logger
    async def _process_images(self, downloaded_images: List[Tuple[int, np.ndarray, str]]) -> List[Tuple[str, np.ndarray]]:
        """
        Асинхронная обработка изображений.
        """
        if not downloaded_images:
            return []
        
        all_results = []
        
        # Используем ProcessPoolExecutor вместо ThreadPoolExecutor для CPU-bound операций
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(downloaded_images))) as executor:
            loop = asyncio.get_event_loop()
            
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
            for idx, result in zip(range(len(downloaded_images)), results):
                if isinstance(result, Exception):
                    print(f"[ERROR] Error processing image {idx}: {str(result)}")
                else:
                    all_results.extend(result)
                    print(f"[PROCESS] Image {idx} processing completed")
        
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
            print(f"[SAVE] Saved {filename}")
        else:
            print(f"[ERROR] Failed to encode image {filename}")

    @async_time_logger
    async def _save_all_images(self, images_to_save: List[Tuple[str, np.ndarray]]) -> None:
        """
        Асинхронно сохраняет все изображения.
        """
        if not images_to_save:
            return
            
        # Создаем задачи для асинхронного сохранения
        tasks = []
        for filename, img_array in images_to_save:
            task = asyncio.create_task(self._save_single_image(filename, img_array))
            tasks.append(task)
        
        # Ждем завершения всех задач сохранения
        await asyncio.gather(*tasks, return_exceptions=True)

    @async_time_logger
    async def process_and_save(self) -> None:
        """
        Основной метод для асинхронной обработки и сохранения изображений.
        """
        total_start_time = time.time()
        
        print("=" * 50)
        print(f"Starting async processing of {self._limit} images")
        print(f"Max concurrent downloads: {self._max_concurrent_downloads}")
        print(f"Output directory: {self._output_dir}")
        print("=" * 50)
        
        try:
            # 1. Получаем URL изображений
            print("\n[STEP 1] Fetching image URLs...")
            await self._fetch_image_urls()
            
            # 2. Асинхронно загружаем изображения
            print(f"\n[STEP 2] Downloading {len(self._image_data)} images asynchronously...")
            downloaded_images = await self._download_images()
            
            if not downloaded_images:
                print("[WARNING] No images downloaded, exiting...")
                return
            
            # 3. Параллельно обрабатываем изображения
            print(f"\n[STEP 3] Processing {len(downloaded_images)} images in parallel...")
            images_to_save = await self._process_images(downloaded_images)
            
            if not images_to_save:
                print("[WARNING] No images processed, exiting...")
                return
            
            # 4. Асинхронно сохраняем все изображения
            print(f"\n[STEP 4] Saving {len(images_to_save)} images asynchronously...")
            await self._save_all_images(images_to_save)
            
        except Exception as e:
            print(f"[CRITICAL ERROR] Processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            
            print("\n" + "=" * 50)
            print(f"Processing completed!")
            print(f"Total time: {total_time:.2f} секунд")
            print(f"Images processed: {len(downloaded_images) if 'downloaded_images' in locals() else 0}")
            print(f"Files saved: {len(images_to_save) if 'images_to_save' in locals() else 0}")
            if 'downloaded_images' in locals() and downloaded_images:
                print(f"Average time per image: {total_time / len(downloaded_images):.2f} секунд")
            print("=" * 50)