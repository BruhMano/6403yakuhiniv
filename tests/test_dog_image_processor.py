"""
Тесты для класса DogImageProcessor.
"""
import unittest
import tempfile
import shutil
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from implementation.dog_image_processing import DogImageProcessor

class TestDogImageProcessorAsync(unittest.IsolatedAsyncioTestCase):
    """Асинхронные тесты с использованием IsolatedAsyncioTestCase."""
    
    def setUp(self):
        """Подготовка тестовых данных."""
        self.temp_dir = tempfile.mkdtemp()
        self.api_key = "test_api_key"
        self.output_dir = f"{self.temp_dir}/output"
        self.grey = False
        self.limit = 2

        # Создаем тестовое цветное изображение
        self.color_image = np.zeros((30, 30, 3), dtype=np.uint8)
        self.color_image[:, :] = [100, 150, 200]
    
    def tearDown(self):
        """Очистка после тестов."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_fetch_image_urls_async(self):
        """Тест асинхронного получения URL изображений."""
        with patch('aiohttp.ClientSession.get') as mock_session:
            # Настраиваем моки
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[
                {
                    "url": "https://test.url/image1.jpg",
                    "breeds": [{"name": "Test Breed 1"}]
                }
            ])
            mock_response.raise_for_status = Mock()
            
            mock_session.return_value.__aenter__.return_value = mock_response
            
            processor = DogImageProcessor(
                api_key=self.api_key,
                output_dir=self.output_dir,
                grey=self.grey,
                limit=self.limit
            )
            
            await processor._fetch_image_urls()
            
            # Проверяем результат
            self.assertEqual(len(processor._image_data), 1)
            self.assertEqual(processor._image_data[0][2], "https://test.url/image1.jpg")
    
    async def test_save_single_image_async(self):
        """Тест асинхронного сохранения изображения."""
        import numpy as np
        
        with patch('aiofiles.open') as mock_open, \
             patch('cv2.imencode') as mock_imencode:
            
            # Настраиваем моки
            mock_file = AsyncMock()
            mock_file.write = AsyncMock()
            mock_file.__aenter__.return_value = mock_file
            mock_open.return_value = mock_file
            
            mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
            
            processor = DogImageProcessor(
                api_key=self.api_key,
                output_dir=self.output_dir,
                grey=self.grey,
                limit=self.limit
            )
            
            test_image = np.zeros((10, 10, 3), dtype=np.uint8)
            
            await processor._save_single_image("test.png", test_image)
            
            # Проверяем что методы были вызваны
            mock_imencode.assert_called_once()
            mock_open.assert_called_once()

    def test_process_single_image_sync_color(self):
        """Тест синхронной обработки цветного изображения."""
        processor = DogImageProcessor(
            api_key=self.api_key,
            output_dir=self.output_dir,
            grey=False,  # цветной режим
            limit=self.limit
        )
        
        # Вызываем метод
        results = processor._process_single_image_sync(
            idx=1,
            img_array=self.color_image,
            breed="test_breed"
        )
        
        # Проверяем результаты
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)  # original, with_edges, with_edges_cv2
        
        # Проверяем каждый результат
        for filename, img_array in results:
            # Проверяем имя файла
            self.assertIsInstance(filename, str)
            self.assertTrue(filename.startswith("1_test_breed_"))
            self.assertTrue(filename.endswith(".png"))
            
            # Проверяем изображение
            self.assertIsInstance(img_array, np.ndarray)
            self.assertEqual(img_array.shape, (30, 30, 3))  # Цветное изображение RGB
            
            # Проверяем диапазон значений
            self.assertTrue(np.all(img_array >= 0))
            self.assertTrue(np.all(img_array <= 255))

if __name__ == "__main__":
    unittest.main()