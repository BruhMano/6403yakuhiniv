"""
Запуск всех тестов проекта.
"""
import unittest
import os


def run_all_tests():
    """Запускает все тесты в проекте."""
    # Находим все тестовые модули
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Загружаем тесты
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Возвращаем код завершения
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    run_all_tests()