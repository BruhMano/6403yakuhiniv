"""
Модуль image_processing.py

Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

Содержит класс ImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""

import cv2

import interfaces

import numpy as np
from numba import njit
import time

class ImageProcessing(interfaces.IImageProcessing):
    """
    Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ, углов и окружностей.

    Методы:
        _convolution(image, kernel): Выполняет свёртку изображения с ядром.
        _rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        _gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Canny).
        corner_detection(image): Обнаруживает углы (Harris).
        circle_detection(image): Обнаруживает окружности (HoughCircles).
    """

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица).

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """

        if image.ndim not in (2, 3):
            raise ValueError("Изображение должно быть grayscale (H, W) или RGB (H, W, C).")
        if kernel.ndim != 2:
            raise ValueError("Ядро должно быть 2D.")
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError("Размеры ядра должны быть нечётными.")
        
        kernel = np.flip(kernel)
        
        h_img, w_img = image.shape[:2]
        h_ker, w_ker = kernel.shape
        n_channels = image.shape[2] if image.ndim == 3 else 1
        pad_h = h_ker // 2
        pad_w = w_ker // 2
        
        if n_channels > 1:
            padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0,0)))
            @njit
            def _rgb_convolution_processing():
                out = np.zeros((h_img, w_img, n_channels))
                for c in range(n_channels):
                    for i in range(h_img):
                        for j in range(w_img):
                            region = padded_img[i:i + h_ker, j:j + w_ker, c]
                            out[i, j, c] = np.sum(region * kernel)
                return out
            output = _rgb_convolution_processing()
        else:
            padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)))
            @njit
            def _grayscale_convolution_processing():
                out = np.zeros((h_img, w_img))
                for i in range(h_img):
                    for j in range(w_img):
                        region = padded_img[i:i + h_ker, j:j + w_ker]
                        out[i, j] = np.sum(region * kernel)
                return out
            output = _grayscale_convolution_processing()
        return output

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Использует функцию cv2.cvtColor для преобразования цветного изображения
        в чёрно-белое.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        return (0.299 * image[:, :, 0] +0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(image.dtype)

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Коррекция осуществляется с помощью таблицы преобразования значений пикселей.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        return np.clip((image/255)**gamma*255 ,0,255).astype(image.dtype)
        # table = np.array([(i / 255.0) ** inv_gamma * 255
        #                   for i in range(256)]).astype("uint8")
        # return cv2.LUT(image, table)

    def _Sobel_gradient(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vertical_Sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        horizontal_Sobel_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        gradient_y = self._convolution(image, vertical_Sobel_filter)
        gradient_x = self._convolution(image, horizontal_Sobel_filter)

        return gradient_y, gradient_x


    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Кэнни (cv2.Canny) для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """

        gray = self._rgb_to_grayscale(image)

        vertical_gradient, horizontal_gradient = self._Sobel_gradient(gray)

        edges = (vertical_gradient**2 + horizontal_gradient**2)**0.5
        
        # edges = cv2.Canny(gray, 100, 200)
        v = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        h = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        return (v**2 + h**2)**0.5

    def corner_detection(self, image: np.ndarray, offset: int, k = 0.04) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Использует алгоритм Харриса (cv2.cornerHarris) для поиска углов.
        Углы выделяются красным цветом на копии исходного изображения.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными углами (красные точки).
        """
        gray = self._rgb_to_grayscale(image)
        gray = np.float32(gray)

        Iy, Ix = self._Sobel_gradient(gray)

        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        h, w = Ix.shape
        dst = np.zeros((h, w), dtype=np.float32)

        for y in range(offset, h - offset):
            for x in range(offset, w - offset):
                Sxx = np.sum(Ixx[y-offset:y+offset+1, x-offset:x+offset+1])
                Syy = np.sum(Iyy[y-offset:y+offset+1, x-offset:x+offset+1])
                Sxy = np.sum(Ixy[y-offset:y+offset+1, x-offset:x+offset+1])

                det = Sxx * Syy - Sxy**2
                trace = Sxx + Syy
                dst[y, x] = det - k * (trace ** 2)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        result = image.copy()
        result[dst > 0.01 * dst.max()] = [255, 0, 0]
        return result

    def circle_detection(self, image: np.ndarray, min_radius = 50, max_radius = 70) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Использует преобразование Хафа (cv2.HoughCircles) для поиска окружностей.
        Найденные окружности выделяются зелёным цветом, центры — красным.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        gray = self._rgb_to_grayscale(image)
        edges = cv2.Canny(gray, 200, 100)
        h, w = edges.shape
        threshold = 0.6
        accumulator = np.zeros((h, w, max_radius - min_radius + 1), dtype=np.int32)

        theta = np.arange(0, 2 * np.pi, 0.1)
        radius_range = np.arange(min_radius, max_radius + 1, 1)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        for i in range(len(radius_range)):
            y_idxs, x_idxs = np.where(edges > 0)
            for j in range(len(x_idxs)):
                x, y = x_idxs[j], y_idxs[j]
                a = np.round(x - radius_range[i] * cos_t).astype(int)
                b = np.round(y - radius_range[i] * sin_t).astype(int)
                valid = (a >= 0) & (a < w) & (b >= 0) & (b < h)
                a, b = a[valid], b[valid]
                for k in range(len(a)):
                    accumulator[b[k], a[k], i] += 1
        max_votes = accumulator.max()
        if max_votes == 0:
            return []
        
        centers = np.where(accumulator >= threshold * max_votes)
        circles = list(zip(centers[1], centers[0], radius_range[centers[2]]))  # (x, y, r)

        # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 75,
        #                        param1=200, param2=30,
        #                        minRadius=50, maxRadius=100)
        result = image.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles:
                center = (i[0], i[1])
                radius = i[2]
                print(radius)
                cv2.circle(result, center, radius, (255, 0, 255), 3)
        return result
