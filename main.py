"""
main.py

Модуль предоставляет интерфейс командной строки для настройки и запуска
процесса загрузки, обработки и сохранения изображений собак из Dog API.

Основные возможности:
    - Загрузка изображений через Dog API с аутентификацией
    - Настройка количества обрабатываемых изображений
    - Выбор режима обработки (цветной/grayscale)
    - Указание директории для сохранения результатов
    - Интеграция с классами DogImage и DogImageProcessor

Запуск:
    python main.py [-l количество изображений] [-grey обработка в одноканальном режиме] [-odir путь_для_сохранения]

Аргументы:
    -l [int]: Количество изображений для извлечения и обработки,
    -odir, --output_dir [str]: Директория для сохранения результата (по умолчанию: result/),
    -grey: Флаг обработки greyscale изображений.

Пример:
    python main.py 
    python main.py -l 10 -odir 'res/' -grey

Автор: Якухин Иван
"""

import argparse
from dotenv import load_dotenv
from implementation import DogImageProcessor
import asyncio
from os import getenv
from log_conf import setup_logging

async def async_main() -> None:
    load_dotenv()
    api_key = getenv('API_KEY')

    logger = setup_logging(log_file="logs/app.log")
    logger.info("Запуск приложения Dog Image Processor")

    parser = argparse.ArgumentParser(
        description="Обработка изображений собак с помощью классов DogImage и DogImageProcessor (Manual & OpenCV).",
    )
    parser.add_argument(
        "-grey",
        action="store_true",
        default=False,
        help="Флаг обработки greyscale изображений",
    )
    parser.add_argument(
        "-odir", "--output_dir",
        default="results/",
        help="Директория для сохранения результата (по умолчанию: result/)",
    )
    parser.add_argument(
        "-l", "--limit",
        default=1,
        help="Количество изображений для извлечения и обработки",
    )

    args = parser.parse_args()

    try:
        # Создаем и запускаем процессор
        processor = DogImageProcessor(api_key, args.output_dir, args.grey, args.limit)
        await processor.process_and_save()
        
        logger.info("Приложение успешно завершило работу")
        
    except KeyboardInterrupt:
        logger.info("Приложение прервано пользователем")
    except Exception as e:
        logger.critical(f"Критическая ошибка в основном потоке: {str(e)}", exc_info=True)
        raise


def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
