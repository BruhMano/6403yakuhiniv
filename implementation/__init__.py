"""
Пакет implementations - различные реализации обработки изображений.

Содержит:
- image_processing: Собственная реализация алгоритмов
- cv2_image_processing: Реализация на основе OpenCV
"""

from .image_processing import ImageProcessing
from .cv2_image_processing import Cv2ImageProcessing
from .dog_image_processing import DogImageProcessor
from .dog_image import ColorDogImage, GrayscaleDogImage

__all__ = ['ImageProcessing', 'Cv2ImageProcessing', 'DogImageProcessor', 'ColorDogImage', 'GrayscaleDogImage']
