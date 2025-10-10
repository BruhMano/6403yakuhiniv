"""
main.py

Пример лабораторной работы по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений с помощью библиотеки OpenCV.
Реализован консольный интерфейс для применения различных методов обработки к изображению:
- обнаружение границ (edges)
- обнаружение углов (corners)
- обнаружение окружностей (circles)

Запуск:
    python main.py <метод> <путь_к_изображению> [-o путь_для_сохранения]

Аргументы:
    метод: edges | corners | circles
    путь_к_изображению: путь к входному изображению
    -o, --output: путь для сохранения результата (по умолчанию: <имя_входного_файла>_result.png)

Пример:
    python main.py edges input.jpg
    python main.py corners input.jpg -o corners_result.png

Автор: Якухин Иван
"""

import argparse
import os
import time
import numpy as np
import cv2

from implementation import ImageProcessing, Cv2ImageProcessing


def timeit(func: callable, image: np.ndarray) -> any:
    """
        Измеряет время выполнения функции и возвращает результат и время.

        Args:
            func: Функция для измерения времени выполнения
            arguments: Позиционные аргументы для функции

        Returns:
            Результат выполнения функции
    """
    start = time.time()
    result = func(image)
    end = time.time()
    print(f"Execution time: {end - start}")

    return result

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing (OpenCV).",
    )
    parser.add_argument(
        "method",
        choices=[
            "edges",
            "corners",
            "circles",
        ],
        help="Метод обработки: edges, corners, circles",
    )
    parser.add_argument(
        "-cv2",
        action="store_true",
        default=False,
        help="Флаг запуска метода из библиотеки cv2",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата (по умолчанию: <input>_result.png)",
    )

    args = parser.parse_args()

    # Загрузка изображения
    image = cv2.imread(args.input)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    if args.cv2:
        processor = Cv2ImageProcessing()
    else:
        processor = ImageProcessing()

    # Выбор метода
    if args.method == "edges":
        result = timeit(processor.edge_detection, image)
    elif args.method == "corners":
        result = timeit(processor.corner_detection, image)
    elif args.method == "circles":
        result = timeit(processor.circle_detection, image)
    else:
        print("Ошибка: неизвестный метод")
        return

    # Определение пути для сохранения
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_result.png"
        if args.cv2:
            output_path = "cv2_" + output_path

    # Сохранение результата
    cv2.imwrite(output_path, result)
    print(f"Результат сохранён в {output_path}")


if __name__ == "__main__":
    main()
