"""
Пакет implementations - различные реализации обработки изображений.

Содержит:
- image_processing: Собственная реализация алгоритмов
- cv2_image_processing: Реализация на основе OpenCV
"""
from .dog_image import ColorDogImage, GrayscaleDogImage
from .cv2_image_processing import Cv2ImageProcessing
from .dog_image_processing import DogImageProcessor
from .image_processing import ImageProcessing
from log_conf import setup_logging

setup_logging(log_file="logs/app.log")

