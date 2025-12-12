"""
Тесты для классов DogImage, ColorDogImage, GrayscaleDogImage.
"""
import unittest
import numpy as np

from implementation.dog_image import ColorDogImage, GrayscaleDogImage
from implementation import ImageProcessing


class TestDogImage(unittest.TestCase):
    """Базовый тестовый класс для DogImage."""
    
    def setUp(self):
        """Подготовка тестовых данных перед каждым тестом."""
        # Создаем тестовое RGB изображение 10x10
        self.rgb_image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Красный квадрат в левом верхнем углу
        self.rgb_image[0:5, 0:5] = [255, 0, 0]
        # Зеленый квадрат в правом верхнем углу
        self.rgb_image[0:5, 5:10] = [0, 255, 0]
        # Синий квадрат в левом нижнем углу
        self.rgb_image[5:10, 0:5] = [0, 0, 255]
        # Серый квадрат в правом нижнем углу
        self.rgb_image[5:10, 5:10] = [128, 128, 128]
        
        # Создаем тестовое grayscale изображение
        self.gray_image = np.zeros((10, 10), dtype=np.uint8)
        for i in range(10):
            for j in range(10):
                self.gray_image[i, j] = (i + j) * 10
        
        self.breed = "test_breed"
        self.url = "http://test.url"
        self.processor = ImageProcessing()


class TestColorDogImage(TestDogImage):
    """Тесты для класса ColorDogImage."""
    
    def test_color_to_grayscale_conversion(self):
        """Тест преобразования цветного изображения в grayscale."""
        dog = ColorDogImage(self.rgb_image, self.breed, self.url, self.processor)
        grayscale = GrayscaleDogImage(dog.image, dog.breed, dog.url, dog.processor)
        
        self.assertIsInstance(grayscale, GrayscaleDogImage)
        self.assertEqual(grayscale.breed, self.breed)
        self.assertEqual(grayscale.url, self.url)
        
        # Проверяем, что grayscale изображение имеет правильную форму
        self.assertEqual(grayscale.image.shape, (10, 10))
        
        # Проверяем, что значения пикселей в диапазоне 0-255
        self.assertTrue(np.all(grayscale.image >= 0))
        self.assertTrue(np.all(grayscale.image <= 255))
    
    def test_edges_detection_custom(self):
        """Тест детекции границ собственными методами."""
        dog = ColorDogImage(self.rgb_image, self.breed, self.url, ImageProcessing())
        edges = GrayscaleDogImage(dog.edges(), dog.breed, dog.url, dog.processor)
        
        self.assertIsInstance(edges, GrayscaleDogImage)
        self.assertEqual(edges.image.shape, (10, 10))
        
        self.assertIsInstance(edges, GrayscaleDogImage)
        self.assertEqual(edges.image.shape, (10, 10))
    
    def test_image_addition(self):
        """Тест сложения двух изображений."""
        dog1 = ColorDogImage(self.rgb_image, self.breed, self.url, self.processor)
        dog2 = GrayscaleDogImage(self.gray_image, self.breed, self.url, self.processor)
        
        result = dog1 + dog2
        
        self.assertIsInstance(result, ColorDogImage)
        self.assertEqual(result.image.shape, self.rgb_image.shape)
        
        # Проверяем, что сложение работает (значения пикселей удвоились, но обрезались до 255)
        gray_to_rgb_image = np.stack([self.gray_image, self.gray_image, self.gray_image], axis=-1)
        expected = np.clip(self.rgb_image.astype(np.int16) + gray_to_rgb_image.astype(np.int16), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result.image, expected)

if __name__ == '__main__':
    unittest.main()