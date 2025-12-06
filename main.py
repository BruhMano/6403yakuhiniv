"""
main.py

Модуль для анализа глобальных выбросов парниковых газов.
Обрабатывает данные о выбросах CO2, CH4, N2O по странам и годам,
вычисляет агрегированную статистику и визуализирует результаты.

Автор: Якухин Иван
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import Generator

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
        yield chunk[columns].copy()


def country_agg(chunks: Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
    """
    Агрегирует данные о выбросах по странам.
    
    Returns:
        Generator: (country_name, list_of_emissions, list_of_emissions_per_capita)
    """
    country_data = pd.DataFrame()
    for chunk in chunks:
        chunk["Emission.Total.Square"] = chunk["Emission.Total"]**2
        chunk_grouped = chunk.groupby("Country.Name").agg({
                                      "Emission.Total": ["sum", "count"], 
                                      "Emission.Total.Per.Capita": ["sum"], 
                                      "Emission.Total.Square": ["sum"]
                                      })
        if country_data.empty:
            country_data = chunk_grouped
        else:
            #country_data = pd.concat([country_data, chunk_grouped]).groupby(level=0).sum()
            country_data = country_data.add(chunk_grouped, fill_value=0)

    return country_data 

def yearly_agg(chunks: Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
    """
    Агрегирует данные о выбросах по годам.
    
    Returns:
        Generator: (year, total_gdp, total_emissions)
    """
    yearly_data = pd.DataFrame()
    
    for chunk in chunks:
        chunk_grouped = chunk.groupby("Year").agg({
                                      "Emission.Total": ["sum"], 
                                      "Country.GDP": ["sum"]
                                      })
        
        if yearly_data.empty:
            yearly_data = chunk_grouped
        else:
            yearly_data = pd.concat([yearly_data, chunk_grouped]).groupby(level=0).sum()
    
    return yearly_data

def avg_emission_per_capita(country_data: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет средние выбросы на душу населения для каждой страны и добавляет к данным.
    
    Args:
        country_data: Агрегированные данные по странам
        
    Returns:
        pd.DataFrame: данные с средним значением
    """
    country_data["Emission.Total.Per.Capita.Mean"] = country_data["Emission.Total.Per.Capita"]['sum'] / country_data["Emission.Total"]['count']
    return country_data


def emission_stats(country_data: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет статистику выбросов: стандартное отклонение и доверительный интервал.
    
    Args:
        country_data: Агрегированные данные по странам
        
    Returns:
       Generator[tuple]: Генератор кортежей (название_страны, mean_выбросов, var_выбросов, доверительный_интервал)
    """
    country_data["Emission.Total.Mean"] = country_data["Emission.Total"]["sum"] / country_data["Emission.Total"]['count']
    country_data["Emission.Total.Var"] = country_data["Emission.Total.Square"]['sum'] / country_data["Emission.Total"]['count'] - country_data["Emission.Total.Mean"]**2
    country_data["Emission.Total.Ci"] = (country_data['Emission.Total.Var']/ country_data["Emission.Total"]['count'])**0.5 * 1.96
    return country_data



def greenest_and_dirtiest(country_data: pd.DataFrame) -> tuple:
    """
    Находит 3 самые "зеленые" и 3 самые "грязные" страны по выбросам на душу населения.
    
    Args:
        country_data: Агрегированные данные по странам
        
    Returns:
        tuple: Кортеж из двух списков:
            - greenest: список из 3 самых "зеленых" стран
            - dirtiest: список из 3 самых "грязных" стран
    """
    country_data = country_data.sort_values("Emission.Total.Per.Capita.Mean")
    greenest = country_data.head(3)
    dirtiest = country_data.tail(3)
    return greenest, dirtiest


def highest_and_lowest_emission_var(country_data: pd.DataFrame) -> tuple:
    """
    Находит страны с наибольшим и наименьшим разбросом выбросов.
    
    Args:
        country_data: Агрегированные данные по странам
        
    Returns:
        tuple: Кортеж из двух списков:
            - lowest_std: 3 страны с наименьшим стандартным отклонением
            - highest_std: 3 страны с наибольшим стандартным отклонением
    """
    sorted_data = country_data.sort_values("Emission.Total.Var")
    lowest_var = sorted_data.head(3)
    highest_var = sorted_data.tail(3)
    return lowest_var, highest_var

def calculate_moving_average(yearly_data: pd.DataFrame, window = 3) -> pd.DataFrame:
    """
    Вычисляет скользящее среднее для годовых данных.
    
    Args:
        yearly_data: Агрегированные данные по годам
        window: Размер окна для скользящего среднего
    
    Returns:
        List: Кортежи (year, gdp, emissions, gdp_ma, emissions_ma)
    """
    yearly_data[['Country.GDP.MA', 'Emission.Total.MA']] = yearly_data[['Country.GDP', 'Emission.Total']].rolling(window).mean()
    return yearly_data


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

    greenest, dirtiest = greenest_and_dirtiest(avg_emission_per_capita(country_agg(emission_calculation(read_chunks("global_emissions.csv" ,chunksize)))))
    least_var, most_var = highest_and_lowest_emission_var(emission_stats(country_agg(emission_calculation(read_chunks("global_emissions.csv" ,chunksize)))))
    gdp_and_emission = calculate_moving_average(yearly_agg(emission_calculation(read_chunks("global_emissions.csv" ,chunksize))))
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    # График 1: Самые "зеленые" страны
    axs[0, 0].bar(greenest.index, greenest["Emission.Total.Per.Capita.Mean"], color='green')
    axs[0, 0].set_title('3 самые «зелёные» страны')
    axs[0, 0].set_ylabel('Выбросы на душу населения (кг/чел)')

    # График 2: Самые "грязные" страны
    axs[1, 0].bar(dirtiest.index, dirtiest["Emission.Total.Per.Capita.Mean"], color='red')
    axs[1, 0].set_title('3 самые «грязные» страны')
    axs[1, 0].set_ylabel('Выбросы на душу населения (кг/чел)')

    # График 3: Страны с наименьшим разбросом выбросов
    axs[0, 1].bar(
        least_var.index,
        least_var["Emission.Total.Mean"],
        yerr=least_var["Emission.Total.Ci"],
        capsize=5, color='lightblue'
    )
    axs[0, 1].set_title('Наименьший разброс')
    axs[0, 1].set_ylabel('Среднее значение выбросов')

    # График 4: Страны с наибольшим разбросом выбросов
    axs[1, 1].bar(
        most_var.index,
        most_var["Emission.Total.Mean"],
        yerr=most_var["Emission.Total.Ci"],
        capsize=5, color='lightblue'
    )
    axs[1, 1].set_title('Наибольший разброс')
    axs[1, 1].set_ylabel('Среднее значение выбросов')

    # График 5: Динамика ВВП
    axs[0, 2].set_title('Динамика ВВП')
    axs[0, 2].plot(gdp_and_emission.index,
                   gdp_and_emission["Country.GDP"], 
                   color='blue', label='ВВП')
    axs[0, 2].plot(gdp_and_emission.index,
                   gdp_and_emission["Country.GDP.MA"],
                   color='blue', linestyle='--', label='ВВП (скользящее среднее)')
    axs[0, 2].set_ylabel('ВВП')
    axs[0, 2].legend(loc='upper left')

    # График 6: Динамика выбросов
    axs[1, 2].set_title('Динамика количества выбросов')
    axs[1, 2].plot(gdp_and_emission.index,
                   gdp_and_emission["Emission.Total"],
                   color='red', label='Выбросы')
    axs[1, 2].plot(gdp_and_emission.index,
                   gdp_and_emission["Emission.Total.MA"],
                   color='red', linestyle='--', label='Выбросы (скользящее среднее)')
    axs[1, 2].set_ylabel('Выбросы')
    axs[1, 2].legend(loc='upper left')

    plt.show()


if __name__ == "__main__":
    main(int(sys.argv[1]))