import numpy as np
import cv2
from dog_image_processor.interfaces import IImageProcessing
from abc import ABC


class DogImage(ABC):
    """
    Абстрактный базовый класс для представления изображения собаки.

    Инкапсулирует изображение, метаданные (порода, URL) и методы обработки.
    Поддерживает арифметические операции (сложение и вычитание) с другими изображениями.
    Является основой для конкретных реализаций: цветных и grayscale-изображений.

    Attributes:
        _image (np.ndarray): Массив пикселей изображения.
        _breed (str): Название породы собаки.
        _url (str): URL исходного изображения.
        _processor (IImageProcessing): Объект, реализующий методы обработки изображений.

    Raises:
        TypeError: При некорректных типах в арифметических операциях.
        cv2.error: При ошибках обработки изображений OpenCV.
    """

    def __init__(self, image_array: np.ndarray, breed: str, url: str, processor: IImageProcessing):
        """
        Инициализирует объект DogImage.

        Args:
            image_array (np.ndarray): Изображение в виде numpy-массива.
            breed (str): Название породы собаки.
            url (str): URL исходного изображения.
            processor (IImageProcessing): Объект-процессор для выполнения операций обработки.
        """
        self._image = image_array
        self._breed = breed
        self._url = url
        self._processor = processor

    @property
    def image(self) -> np.ndarray:
        """np.ndarray: Возвращает изображение в виде numpy-массива."""
        return self._image
    
    @image.setter
    def image(self, new_img: np.ndarray) -> None:
        """
        Устанавливает новое изображение.

        Args:
            new_img (np.ndarray): Новый массив пикселей изображения.
        """
        self._image = new_img

    @property
    def breed(self) -> str:
        """str: Возвращает название породы собаки."""
        return self._breed

    @property
    def url(self) -> str:
        """str: Возвращает URL исходного изображения."""
        return self._url

    @property
    def processor(self) -> IImageProcessing:
        """IImageProcessing: Возвращает объект-процессор обработки изображений."""
        return self._processor
    
    @processor.setter
    def processor(self, new_proc: IImageProcessing) -> None:
        """
        Устанавливает новый процессор обработки изображений.

        Args:
            new_proc (IImageProcessing): Новый объект-процессор.
        """
        self._processor = new_proc

    def _convolution(self, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет свёртку текущего изображения с заданным ядром.

        Args:
            kernel (np.ndarray): Ядро свёртки.

        Returns:
            np.ndarray: Результат свёртки.

        Note:
            Использует реализацию свёртки из установленного процессора.
        """
        return self.processor._convolution(self.image, kernel)

    def _rgb_to_grayscale(self) -> np.ndarray:
        """
        Преобразует текущее изображение в оттенки серого (если оно цветное).

        Returns:
            np.ndarray: Изображение в оттенках серого.

        Note:
            Для grayscale изображений возвращает копию исходного массива.
        """
        return self.processor._rgb_to_grayscale(self.image)

    def _gamma_correction(self, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к текущему изображению.

        Args:
            gamma (float): Параметр гамма-коррекции. 
                Значения < 1 усиливают тёмные тона, > 1 - светлые.

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        return self.processor._gamma_correction(self.image, gamma)

    def edges(self) -> np.ndarray:
        """
        Выполняет обнаружение границ на текущем изображении.

        Returns:
            np.ndarray: Изображение с выделенными границами.

        Note:
            Использует алгоритм обнаружения границ из установленного процессора.
        """
        return self.processor.edge_detection(self.image)

    def corners(self) -> np.ndarray:
        """
        Выполняет обнаружение углов на текущем изображении.

        Returns:
            np.ndarray: Изображение с выделенными углами.

        Note:
            Использует алгоритм обнаружения углов из установленного процессора.
        """
        return self.processor.corner_detection(self.image)

    def circles(self) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на текущем изображении.

        Returns:
            np.ndarray: Изображение с выделенными окружностями.

        Note:
            Использует алгоритм обнаружения окружностей из установленного процессора.
        """
        return self.processor.circle_detection(self.image)

    def save(self, filename: str) -> None:
        """
        Сохраняет изображение в файл.

        Args:
            filename (str): Путь и имя файла для сохранения.

        Raises:
            cv2.error: При ошибках записи файла OpenCV.
            OSError: При проблемах с файловой системой.
        """
        cv2.imwrite(filename, self.image)

    def __add__(self, other: "DogImage") -> "DogImage":
        """
        Складывает текущее изображение с другим изображением.

        Размер второго изображения автоматически приводится к размеру текущего.
        Тип результата совпадает с типом текущего объекта (первого операнда).
        Операция не коммутативна: Grayscale + Color → Grayscale,
        Color + Grayscale → Color (благодаря broadcasting).

        Args:
            other (DogImage): Другое изображение для сложения.

        Returns:
            DogImage: Новое изображение — результат сложения.

        Raises:
            TypeError: Если other не является экземпляром DogImage.
            cv2.error: При ошибках изменения размера изображения.

        Note:
            Значения пикселей ограничиваются диапазоном [0, 255].
        """
        if not isinstance(other, DogImage):
            raise TypeError("Операция возможна только с объектами типа DogImage")
        other_resized = cv2.resize(other.image, (self.image.shape[1], self.image.shape[0]))
        new_image = np.clip((self.image.astype(np.int16) + other_resized.astype(np.int16)), 0, 255).astype(np.uint8)
        return self.__class__(new_image, self.breed, self.url, self.processor)

    def __sub__(self, other: 'DogImage') -> 'DogImage':
        """
        Вычитает другое изображение из текущего.

        Размер второго изображения автоматически приводится к размеру текущего.
        Тип результата совпадает с типом текущего объекта.

        Args:
            other (DogImage): Другое изображение для вычитания.

        Returns:
            DogImage: Новое изображение — результат вычитания.

        Raises:
            TypeError: Если other не является экземпляром DogImage.
            cv2.error: При ошибках изменения размера изображения.

        Note:
            Значения пикселей ограничиваются диапазоном [0, 255].
        """
        if not isinstance(other, DogImage):
            raise TypeError("Операция возможна только с объектами типа DogImage")
        other_resized = cv2.resize(other.image, (self.image.shape[1], self.image.shape[0]))
        new_image = np.clip((self.image.astype(np.int16) - other_resized.astype(np.int16)), 0, 255).astype(np.uint8)
        return self.__class__(new_image, self.breed, self.url, self.processor)

    def __str__(self) -> str:
        """
        Возвращает строковое представление объекта.

        Returns:
            str: Строка вида "DogImage(breed=..., shape=(H, W[, C]))".
        """
        return f"DogImage(breed={self.breed}, shape={self.image.shape})"


class ColorDogImage(DogImage):
    """
    Класс для представления цветного изображения собаки.

    Наследует всю функциональность от DogImage.
    Автоматически преобразует grayscale изображения в цветные при инициализации
    путем дублирования каналов.

    Attributes:
        Наследует все атрибуты от DogImage.

    Note:
        Всегда гарантирует 3-канальное представление изображения (RGB).
    """

    def __init__(self, image_array: np.ndarray, breed: str, url: str, processor: IImageProcessing):
        """
        Инициализирует объект ColorDogImage.

        Если входное изображение grayscale (2D), оно преобразуется в цветное
        путем дублирования каналов.

        Args:
            image_array (np.ndarray): Изображение в виде numpy-массива.
            breed (str): Название породы собаки.
            url (str): URL исходного изображения.
            processor (IImageProcessing): Объект-процессор для выполнения операций обработки.
        """
        if image_array.ndim == 2:
            image_array = np.stack([image_array, image_array, image_array], axis=-1)
        super().__init__(image_array, breed, url, processor)

    def __add__(self, other: DogImage) -> 'ColorDogImage':
        """
        Складывает текущее цветное изображение с другим изображением.

        Если other — grayscale, оно преобразуется в цветное перед сложением
        путем дублирования каналов.

        Args:
            other (DogImage): Другое изображение для сложения.

        Returns:
            ColorDogImage: Новое цветное изображение — результат сложения.

        Raises:
            TypeError: Если other не является экземпляром DogImage.

        Note:
            Сохраняет цветное представление результата.
        """
        if not isinstance(other, DogImage):
            raise TypeError("Операция возможна только с объектами типа DogImage")
        if other.image.ndim == 2:
            other.image = np.stack([other.image, other.image, other.image], axis=-1)
        return super().__add__(other)

    def __sub__(self, other: DogImage) -> 'ColorDogImage':
        """
        Вычитает другое изображение из текущего цветного изображения.

        Если other — grayscale, оно преобразуется в цветное перед вычитанием
        путем дублирования каналов.

        Args:
            other (DogImage): Другое изображение для вычитания.

        Returns:
            ColorDogImage: Новое цветное изображение — результат вычитания.

        Raises:
            TypeError: Если other не является экземпляром DogImage.

        Note:
            Сохраняет цветное представление результата.
        """
        if not isinstance(other, DogImage):
            raise TypeError("Операция возможна только с объектами типа DogImage")
        if other.image.ndim == 2:
            other.image = np.stack([other.image, other.image, other.image], axis=-1)
        return super().__sub__(other)


class GrayscaleDogImage(DogImage):
    """
    Класс для представления изображения собаки в оттенках серого.

    При инициализации автоматически преобразует цветные изображения в grayscale.
    При арифметических операциях с цветными изображениями приводит их к grayscale,
    чтобы сохранить тип результата как grayscale.

    Attributes:
        Наследует все атрибуты от DogImage.

    Note:
        Всегда гарантирует 1-канальное представление изображения.
    """

    def __init__(self, image_array: np.ndarray, breed: str, url: str, processor: IImageProcessing):
        """
        Инициализирует объект GrayscaleDogImage.

        Если входное изображение цветное (3D), оно преобразуется в grayscale.

        Args:
            image_array (np.ndarray): Изображение в виде numpy-массива.
            breed (str): Название породы собаки.
            url (str): URL исходного изображения.
            processor (IImageProcessing): Объект-процессор для выполнения операций обработки.
        """
        if image_array.ndim == 3:
            image_array = processor._rgb_to_grayscale(image_array)
        super().__init__(image_array, breed, url, processor)

    def __add__(self, other: DogImage) -> 'GrayscaleDogImage':
        """
        Складывает текущее grayscale-изображение с другим изображением.

        Если other — цветное, оно преобразуется в grayscale перед сложением.
        Результат всегда имеет тип GrayscaleDogImage.

        Args:
            other (DogImage): Другое изображение для сложения.

        Returns:
            GrayscaleDogImage: Новое grayscale-изображение — результат сложения.

        Raises:
            TypeError: Если other не является экземпляром DogImage.

        Note:
            Сохраняет grayscale представление результата.
        """
        if not isinstance(other, DogImage):
            raise TypeError("Операция возможна только с объектами типа DogImage")
        if other.image.ndim == 3:
            other = GrayscaleDogImage(other.image, other.breed, other.url, other.processor)
        return super().__add__(other)

    def __sub__(self, other: DogImage) -> 'GrayscaleDogImage':
        """
        Вычитает другое изображение из текущего grayscale-изображения.

        Если other — цветное, оно преобразуется в grayscale перед вычитанием.
        Результат всегда имеет тип GrayscaleDogImage.

        Args:
            other (DogImage): Другое изображение для вычитания.

        Returns:
            GrayscaleDogImage: Новое grayscale-изображение — результат вычитания.

        Raises:
            TypeError: Если other не является экземпляром DogImage.

        Note:
            Сохраняет grayscale представление результата.
        """
        if not isinstance(other, DogImage):
            raise TypeError("Операция возможна только с объектами типа DogImage")
        if other.image.ndim == 3:
            other = GrayscaleDogImage(other.image, other.breed, other.url, other.processor)
        return super().__sub__(other)