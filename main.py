"""
main.py

Модуль для анализа глобальных выбросов парниковых газов.
Обрабатывает данные о выбросах CO2, CH4, N2O по странам и годам,
вычисляет агрегированную статистику и визуализирует результаты.

Автор: Якухин Иван
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Generator
from collections import defaultdict, deque

CSV_FILE = 'global_emissions.csv'


def read_chunks(filename: str, chunksize: int = 10000) -> Generator[pd.DataFrame, None, None]:
    """
    Читает CSV-файл порциями (чанками) для обработки больших данных.
    
    Args:
        filename (str): Путь к CSV-файлу с данными о выбросах
        chunksize (int, optional): Размер чанка в строках. По умолчанию 10000.
    
    Yields:
        pd.DataFrame: Очередной чанк данных
    """
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        yield chunk


def emission_calculation(chunks: Generator[pd.DataFrame, None, None]) -> Generator[pd.DataFrame, None, None]:
    """
    Вычисляет общие выбросы и выбросы на душу населения для каждого чанка.
    
    Args:
        chunks (pd.DataFrame): Генератор чанков данных
        
    Yields:
        pd.DataFrame: Чанк с добавленными колонками Emission.Total и 
                     Emission.Total.Per.Capita
    """
    gases = ['Emissions.Production.CO2.Total', 'Emissions.Production.CH4', 'Emissions.Production.N2O']
    columns = ["Year", "Country.Name", "Country.GDP", "Emission.Total", "Emission.Total.Per.Capita"]
    for chunk in chunks:
        chunk['Emission.Total'] = chunk[gases].sum(axis=1)
        chunk['Emission.Total.Per.Capita'] = chunk['Emission.Total'] / chunk['Country.Population'] * 1e6
        yield chunk[columns]


def country_agg(chunks: Generator[pd.DataFrame, None, None]) -> Generator[tuple, None, None]:
    """
    Агрегирует данные о выбросах по странам.
    
    Returns:
        Generator: (country_name, list_of_emissions, list_of_emissions_per_capita)
    """
    country_data = defaultdict(lambda: {'emissions': [], 'emissions_per_capita': []})
    for chunk in chunks:
        for _, row in chunk.iterrows():
            country = row['Country.Name']
            country_data[country]['emissions'].append(row['Emission.Total'])
            country_data[country]['emissions_per_capita'].append(row['Emission.Total.Per.Capita'])

    for country, data in country_data.items():
        yield country, data['emissions'], data['emissions_per_capita']

def yearly_agg(chunks: Generator[pd.DataFrame, None, None]) -> Generator[tuple, None, None]:
    """
    Агрегирует данные о выбросах по годам.
    
    Returns:
        Generator: (year, total_gdp, total_emissions)
    """
    yearly_data = defaultdict(lambda: {'gdp': 0.0, 'emissions': 0.0})
    
    for chunk in chunks:
        for _, row in chunk.iterrows():
            year = row['Year']
            yearly_data[year]['gdp'] += row['Country.GDP']
            yearly_data[year]['emissions'] += row['Emission.Total']
    
    for year in sorted(yearly_data.keys()):
        data = yearly_data[year]
        yield year, data['gdp'], data['emissions']

def avg_emission_per_capita(country_data: Generator[tuple, None, None]) -> Generator[tuple, None, None]:
    """
    Вычисляет средние выбросы на душу населения для каждой страны.
    
    Args:
        country_data: Агрегированные данные по странам
        
    Returns:
        Generator[tuple]: Генератор кортежей (название_страны, средние_выбросы_на_душу_населения)
    """
    return ((country[0], np.mean(country[2])) for country in country_data)


def emission_stats(country_data: Generator[tuple, None, None]) -> Generator[tuple, None, None]:
    """
    Вычисляет статистику выбросов: стандартное отклонение и доверительный интервал.
    
    Args:
        country_data: Агрегированные данные по странам
        
    Returns:
       Generator[tuple]: Генератор кортежей (название_страны, mean_выбросов, var_выбросов, доверительный_интервал)
    """
    return ((country[0], np.mean(country[1]), np.var(country[1]), 
             1.96 * np.std(country[1])/len(country[1])**0.5) 
             for country in country_data)


def greenest_and_dirtiest(country_data: Generator[tuple, None, None]) -> tuple:
    """
    Находит 3 самые "зеленые" и 3 самые "грязные" страны по выбросам на душу населения.
    
    Args:
        country_data: Агрегированные данные по странам
        
    Returns:
        tuple: Кортеж из двух списков:
            - greenest: список из 3 самых "зеленых" стран
            - dirtiest: список из 3 самых "грязных" стран
    """
    avg_per_capita = list(avg_emission_per_capita(country_data))
    sorted_avg_per_capita = sorted(avg_per_capita, key=lambda item: item[1])
    greenest = sorted_avg_per_capita[:3]
    dirtiest = sorted_avg_per_capita[-3:][::-1]
    return greenest, dirtiest


def highest_and_lowest_emission_var(country_data: Generator[tuple, None, None]) -> tuple:
    """
    Находит страны с наибольшим и наименьшим разбросом выбросов.
    
    Args:
        country_data: Агрегированные данные по странам
        
    Returns:
        tuple: Кортеж из двух списков:
            - lowest_std: 3 страны с наименьшим стандартным отклонением
            - highest_std: 3 страны с наибольшим стандартным отклонением
    """
    data = list(emission_stats(country_data))
    sorted_data = sorted(data, key=lambda item: item[2])
    lowest_var = sorted_data[:3]
    highest_var = sorted_data[-3:][::-1]
    return lowest_var, highest_var

def calculate_moving_average(yearly_data: Generator[tuple, None, None], window: int = 3) -> list[tuple]:
    """
    Вычисляет скользящее среднее для годовых данных.
    
    Args:
        yearly_data: Агрегированные данные по годам
        window: Размер окна для скользящего среднего
    
    Returns:
        List: Кортежи (year, gdp, emissions, gdp_ma, emissions_ma)
    """
    result = []

    gdp_buffer = deque(maxlen=window)
    emissions_buffer = deque(maxlen=window)
    year_buffer = deque(maxlen=window)
    
    for year, gdp, emissions in yearly_data:
        gdp_buffer.append(gdp)
        emissions_buffer.append(emissions)
        year_buffer.append(year)
        
        if len(gdp_buffer) == window:
            current_year = year_buffer[-1]
            gdp_ma = sum(gdp_buffer) / window
            emissions_ma = sum(emissions_buffer) / window
            
            result.append((current_year, gdp, emissions, gdp_ma, emissions_ma))
    
    return result


def main(chunksize: int) -> None:
    """
    Основная функция: выполняет анализ данных и визуализирует результаты.
    
    Создает 6 графиков:
    1. 3 самые "зеленые" страны
    2. 3 самые "грязные" страны  
    3. Страны с наименьшим разбросом выбросов
    4. Страны с наибольшим разбросом выбросов
    5. Динамика ВВП по годам
    6. Динамика выбросов по годам
    
    Args:
        chunksize (int): Размер чанка для обработки данных
    """

    greenest, dirtiest = greenest_and_dirtiest(country_agg(emission_calculation(read_chunks("global_emissions.csv" ,chunksize))))
    least_var, most_var = highest_and_lowest_emission_var(country_agg(emission_calculation(read_chunks("global_emissions.csv" ,chunksize))))
    gdp_and_emission = calculate_moving_average(yearly_agg(emission_calculation(read_chunks("global_emissions.csv" ,chunksize))))
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    # График 1: Самые "зеленые" страны
    axs[0, 0].bar([c for c, _ in greenest], [v for _, v in greenest], color='green')
    axs[0, 0].set_title('3 самые «зелёные» страны')
    axs[0, 0].set_ylabel('Выбросы на душу населения (кг/чел)')

    # График 2: Самые "грязные" страны
    axs[1, 0].bar([c for c, _ in dirtiest], [v for _, v in dirtiest], color='red')
    axs[1, 0].set_title('3 самые «грязные» страны')
    axs[1, 0].set_ylabel('Выбросы на душу населения (кг/чел)')

    # График 3: Страны с наименьшим разбросом выбросов
    axs[0, 1].bar(
        [c[0] for c in least_var],
        [v[1] for v in least_var],
        yerr=[v[3] for v in least_var],
        capsize=5, color='lightblue'
    )
    axs[0, 1].set_title('Наименьший разброс')
    axs[0, 1].set_ylabel('Среднее значение выбросов')

    # График 4: Страны с наибольшим разбросом выбросов
    axs[1, 1].bar(
        [c[0] for c in most_var],
        [v[1] for v in most_var],
        yerr=[v[3] for v in most_var],
        capsize=5, color='lightblue'
    )
    axs[1, 1].set_title('Наибольший разброс')
    axs[1, 1].set_ylabel('Среднее значение выбросов')

    # График 5: Динамика ВВП
    axs[0, 2].set_title('Динамика ВВП')
    axs[0, 2].plot([y[0] for y in gdp_and_emission],
                   [g[1] for g in gdp_and_emission], 
                   color='blue', label='ВВП')
    axs[0, 2].plot([y[0] for y in gdp_and_emission],
                   [g[3] for g in gdp_and_emission],
                   color='blue', linestyle='--', label='ВВП (скользящее среднее)')
    axs[0, 2].set_ylabel('ВВП')
    axs[0, 2].legend(loc='upper left')

    # График 6: Динамика выбросов
    axs[1, 2].set_title('Динамика количества выбросов')
    axs[1, 2].plot([y[0] for y in gdp_and_emission],
                   [e[2] for e in gdp_and_emission], 
                   color='red', label='Выбросы')
    axs[1, 2].plot([y[0] for y in gdp_and_emission],
                   [e[4] for e in gdp_and_emission], 
                   color='red', linestyle='--', label='Выбросы (скользящее среднее)')
    axs[1, 2].set_ylabel('Выбросы')
    axs[1, 2].legend(loc='upper left')

    plt.show()


if __name__ == "__main__":
    main(int(sys.argv[1]))