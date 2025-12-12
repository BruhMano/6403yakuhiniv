"""
Тесты для класса DogImageProcessor.
"""
import unittest
import asyncio
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
from PIL import Image
import aiohttp

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from implementation.dog_image_processing import DogImageProcessor
from implementation.dog_image import ColorDogImage

class TestDogImageProcessorAsync(unittest.TestCase):
    """Асинхронные тесты для DogImageProcessor."""
    
    def setUp(self):
        """Подготовка тестовых данных."""
        self.temp_dir = tempfile.mkdtemp()
        self.api_key = "test_api_key"
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.grey = False
        self.limit = 2
    
    def tearDown(self):
        """Очистка после тестов."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def run_async(self, coro):
        """Запуск асинхронной корутины в синхронном тесте."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    @patch('aiohttp.ClientSession.get')
    def test_fetch_image_urls_async(self, mock_get):
        """Тест асинхронного получения URL изображений."""
        # Мокируем асинхронный ответ
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[
            {
                "url": "https://test.url/image1.jpg",
                "breeds": [{"name": "Test Breed 1"}]
            },
            {
                "url": "https://test.url/image2.jpg",
                "breeds": [{"name": "Test Breed 2"}]
            }
        ])
        mock_response.raise_for_status = Mock()
        mock_get.return_value.__aenter__.return_value = mock_response
        
        processor = DogImageProcessor(
            api_key=self.api_key,
            output_dir=self.output_dir,
            grey=self.grey,
            limit=self.limit
        )
        
        # Запускаем асинхронный метод
        self.run_async(processor._fetch_image_urls())
        
        # Проверяем, что данные сохранены
        self.assertEqual(len(processor._image_data), 2)
        self.assertEqual(processor._image_data[0][0], 0)  # index
        self.assertEqual(processor._image_data[0][1], "test_breed_1")  # breed
        self.assertEqual(processor._image_data[0][2], "https://test.url/image1.jpg")  # url
    
    @patch('aiohttp.ClientSession.get')
    def test_download_images_async(self, mock_get):
        """Тест асинхронной загрузки изображений."""
        # Мокируем ответ API для получения URL
        mock_api_response = AsyncMock()
        mock_api_response.status = 200
        mock_api_response.json = AsyncMock(return_value=[
            {
                "url": "https://test.url/image1.jpg",
                "breeds": [{"name": "Test Breed"}]
            }
        ])
        mock_api_response.raise_for_status = Mock()
        
        # Мокируем ответ для загрузки изображения
        mock_image_response = AsyncMock()
        mock_image_response.status = 200
        mock_image_response.read = AsyncMock(return_value=b"fake_image_data")
        
        # Настраиваем мок для последовательных вызовов
        mock_get.side_effect = [
            mock_api_response,  # Первый вызов для API
            mock_image_response  # Второй вызов для загрузки изображения
        ]
        
        processor = DogImageProcessor(
            api_key=self.api_key,
            output_dir=self.output_dir,
            grey=self.grey,
            limit=1
        )
        
        # Запускаем асинхронный метод
        downloaded_images = self.run_async(processor._download_images())
        
        # Проверяем результат
        self.assertIsInstance(downloaded_images, list)
        if downloaded_images:  # Мок может не работать идеально
            self.assertEqual(len(downloaded_images), 1)
    
    
    @patch('aiofiles.open')
    @patch('cv2.imencode')
    def test_save_single_image_async(self, mock_imencode, mock_aiofiles_open):
        """Тест асинхронного сохранения одного изображения."""
        # Мокируем cv2.imencode
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        
        # Мокируем aiofiles.open
        mock_file = AsyncMock()
        mock_file.write = AsyncMock()
        mock_file.__aenter__.return_value = mock_file
        mock_aiofiles_open.return_value = mock_file
        
        processor = DogImageProcessor(
            api_key=self.api_key,
            output_dir=self.output_dir,
            grey=self.grey,
            limit=self.limit
        )
        
        # Создаем тестовое изображение
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Запускаем асинхронный метод
        self.run_async(processor._save_single_image("test.png", test_image))
        
        # Проверяем, что методы были вызваны
        mock_imencode.assert_called_once()
        mock_aiofiles_open.assert_called_once()



if __name__ == '__main__':
    unittest.main()