"""
Модуль image_processing.py

Реализация интерфейса IImageProcessing без использования библиотеки OpenCV.

Содержит класс ImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (Преобразование Хафа)

Модуль предназначен для учебных целей
(лабораторная работа по курсу "Технологии программирования на Python").
"""
import cv2
from dog_image_processor.interfaces import IImageProcessing
import numpy as np
from numba import njit


class ImageProcessing(IImageProcessing):
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

        kernel = np.flip(kernel)

        img_height, img_width = image.shape[:2]
        ker_height, ker_width = kernel.shape
        n_channels = image.shape[2] if image.ndim == 3 else 1
        pad_h = ker_height // 2
        pad_w = ker_width // 2

        if n_channels > 1:
            padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))

            @njit
            def _rgb_convolution_processing() -> np.ndarray:
                """
                Внутренняя функция вычисления свертки для цветного изображения.
                Необходимость для корректной работы декоратора njit.
                """
                out = np.zeros((img_height, img_width, n_channels))
                for chanel in range(n_channels):
                    for row in range(img_height):
                        for col in range(img_width):
                            region = padded_img[row:row + ker_height, col:col + ker_width, chanel]
                            out[row, col, chanel] = np.sum(region * kernel)

                return out

            output = _rgb_convolution_processing()
        else:
            padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)))

            @njit
            def _grayscale_convolution_processing() -> np.ndarray:
                """
                Внутренняя функция вычисления свертки для черно-белого изображения.
                Необходимость для корректной работы декоратора njit.
                """
                out = np.zeros((img_height, img_width))
                for row in range(img_height):
                    for col in range(img_width):
                        region = padded_img[row:row + ker_height, col:col + ker_width]
                        out[row, col] = np.sum(region * kernel)

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
        if image.ndim < 3:
            return image
        return (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2]).astype(image.dtype)

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
        return np.clip((image / 255) ** gamma * 255, 0, 255).astype(image.dtype)

    def _gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """
        Создаёт 2D гауссово ядро.

        Args:
            size (int): Размер ядра (должен быть нечётным, например, 3, 5, 7).
            sigma (float): Стандартное отклонение гауссианы.

        Returns:
            np.ndarray: Нормированное гауссово ядро формы (size, size).
        """
        ax = np.arange(-size // 2 + 1, size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) * (2 * sigma**2))
        return kernel / np.sum(kernel)

    def _sobel_filter(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Применяет оператор Собеля для вычисления градиентов.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            tuple: Градиент по модулю и направление градиента.
        """
        horizontal_sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        vertical_sobel_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        gradient_x = self._convolution(image, vertical_sobel_filter)
        gradient_y = self._convolution(image, horizontal_sobel_filter)

        return gradient_x, gradient_y

    def _non_max_suppression(self, x_grad: np.ndarray, y_grad: np.ndarray, tolerance = 0.9) -> np.ndarray:
        """
        Подавление немаксимумов.

        Args:
            x_grad (np.ndarray): приближение производной по x.
            y_grad (np.ndarray): приближение производной по y.
            tolerance (float): Допуск в отклонении от максимума градиента.

        Returns:
            np.ndarray: Результат подавления немаксимумов.
        """
        grad = (x_grad**2 + y_grad**2)**0.5
        theta = np.arctan2(y_grad, x_grad)
        height, width = grad.shape
        output = np.zeros((height, width), dtype=np.float32)
        angle = theta * 180.0 / np.pi
        angle[angle < 0] += 180

        for row in range(1, height - 1):
            for col in range(1, width - 1):
                neighbor_ahead = 255
                neighbor_behind = 255

                # 0
                if (0 <= angle[row, col] < 22.5) or (157.5 <= angle[row, col] <= 180):
                    neighbor_ahead = grad[row, col + 1]    # Правый сосед
                    neighbor_behind = grad[row, col - 1]   # Левый сосед
                # 45
                elif 22.5 <= angle[row, col] < 67.5:
                    neighbor_ahead = grad[row + 1, col - 1]  # Нижний-левый
                    neighbor_behind = grad[row - 1, col + 1]  # Верхний-правый
                # 90
                elif 67.5 <= angle[row, col] < 112.5:
                    neighbor_ahead = grad[row + 1, col]    # Нижний сосед
                    neighbor_behind = grad[row - 1, col]   # Верхний сосед
                # 135
                elif 112.5 <= angle[row, col] < 157.5:
                    neighbor_ahead = grad[row - 1, col - 1]  # Верхний-левый
                    neighbor_behind = grad[row + 1, col + 1]  # Нижний-правый

                # Если текущий пиксель - локальный максимум в направлении градиента
                if (grad[row, col] >= tolerance*neighbor_ahead) and (grad[row, col] >= tolerance*neighbor_behind):
                    output[row, col] = grad[row, col]  # Сохраняем значение
                else:
                    output[row, col] = 0  # Обнуляем

        return output

    def _hysteresis_thresholding(self, img: np.ndarray, low_thresh: float, high_thresh: float) -> np.ndarray:
        """
        Двухпороговая фильтрация с гистерезисом.

        Args:
            img (np.ndarray): Входное изображение.
            low_thresh (float): Нижний порог.
            high_thresh (float): Верхний порог.

        Returns:
            np.ndarray: Результат фильтрации.
        """
        height, width = img.shape
        res = np.zeros((height, width), dtype=np.uint8)

        # Сильные и слабые границы
        strong = 255
        weak = 50

        res[img >= high_thresh] = strong
        res[(img >= low_thresh) & (img < high_thresh)] = weak

        # Соединение слабых границ с сильными
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                if res[row, col] == weak:
                    if (res[row + 1, col - 1] == strong or res[row + 1, col] == strong or
                            res[row + 1, col + 1] == strong or res[row, col - 1] == strong or
                            res[row, col + 1] == strong or res[row - 1, col - 1] == strong or
                            res[row - 1, col] == strong or res[row - 1, col + 1] == strong):
                        res[row, col] = strong
                    else:
                        res[row, col] = 0

        return res

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Кэнни для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        gray = self._rgb_to_grayscale(image)
        g_kernel = self._gaussian_kernel(5, 1)
        denoised = self._convolution(gray, g_kernel)
        grad_x, grad_y = self._sobel_filter(denoised)
        nms = self._non_max_suppression(grad_x, grad_y)
        edges = self._hysteresis_thresholding(nms, 100, 200)

        return edges

    def corner_detection(self, image: np.ndarray, offset: int = 2, k: float = 0.04) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Использует алгоритм Харриса для поиска углов.
        Углы выделяются красным цветом на копии исходного изображения.

        Args:
            image (np.ndarray): Входное изображение (RGB).
            offset (int): Смещение для окрестности.
            k (float): Параметр алгоритма Харриса.

        Returns:
            np.ndarray: Изображение с выделенными углами (красные точки).
        """
        gray = self._rgb_to_grayscale(image)

        ix, iy = self._sobel_filter(gray)

        ixx = ix * ix
        iyy = iy * iy
        ixy = ix * iy

        height, width = ix.shape
        dst = np.zeros((height, width), dtype=np.float32)

        for row in range(offset, height - offset):
            for col in range(offset, width - offset):
                sxx = np.sum(ixx[row - offset:row + offset + 1, col - offset:col + offset + 1])
                syy = np.sum(iyy[row - offset:row + offset + 1, col - offset:col + offset + 1])
                sxy = np.sum(ixy[row - offset:row + offset + 1, col - offset:col + offset + 1])

                det = sxx * syy - sxy**2
                trace = sxx + syy
                dst[row, col] = det - k * (trace ** 2)

        result = np.zeros((height, width, 3))
        result[dst > 0.01 * dst.max()] = [255, 0, 0]
        return result

    def circle_detection(self, image: np.ndarray, min_radius: int = 50, max_radius: int = 100) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Использует преобразование Хафа для поиска окружностей.
        Найденные окружности выделяются красным цветом.

        Args:
            image (np.ndarray): Входное изображение (RGB).
            min_radius (int): Минимальный радиус окружности.
            max_radius (int): Максимальный радиус окружности.

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        edges = self.edge_detection(image)
        height, width = edges.shape
        threshold = 0.6

        theta = np.arange(0, 2 * np.pi, 0.1)
        radius_range = np.arange(min_radius, max_radius + 1, 1)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        @njit
        def internal_accumulator_processing():
            """
                Внутренняя функция заполнения аккумулятора см. алгоритм преобразования Хафа.
                Необходимость для корректной работы декоратора njit.
            """
            accumulator = np.zeros((height, width, max_radius - min_radius + 1))
            for radius_index in range(len(radius_range)):
                current_radius = radius_range[radius_index]
                edge_y_coords, edge_x_coords = np.where(edges > 0)

                for edge_point_index in range(len(edge_x_coords)):
                    edge_x, edge_y = edge_x_coords[edge_point_index], edge_y_coords[edge_point_index]

                    # Вычисляем возможные центры для данной точки границы и радиуса
                    center_x_candidates = np.round(edge_x - current_radius * cos_theta).astype(np.int64)
                    center_y_candidates = np.round(edge_y - current_radius * sin_theta).astype(np.int64)

                    # Фильтруем центры, выходящие за границы изображения
                    valid_centers_mask = (
                        (center_x_candidates >= 0) &
                        (center_x_candidates < width) &
                        (center_y_candidates >= 0) &
                        (center_y_candidates < height)
                    )
                    valid_center_x = center_x_candidates[valid_centers_mask]
                    valid_center_y = center_y_candidates[valid_centers_mask]

                    # Голосуем за каждый валидный центр
                    for candidate_index in range(len(valid_center_x)):
                        accumulator[valid_center_y[candidate_index],
                                    valid_center_x[candidate_index],
                                    radius_index] += 1
            return accumulator
            
        accumulator = internal_accumulator_processing()
        result = np.zeros((height, width, 3))
        max_votes = accumulator.max()
        if max_votes == 0:
            return result

        centers = np.where(accumulator >= threshold * max_votes)
        circles = list(zip(centers[1], centers[0], radius_range[centers[2]]))  # (x, y, radius)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle_params in circles:
                center_x, center_y, radius = circle_params
                center = (center_x, center_y)
                cv2.circle(result, center, radius, (0, 0, 255), 3)

        return result
