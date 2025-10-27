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

CSV_FILE = 'global_emissions.csv'


def read_chunks(filename: str, chunksize: int = 10000) -> pd.DataFrame:
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


def emission_calculation(chunks: pd.DataFrame) -> pd.DataFrame:
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


def country_and_yearly_stats_factory(chunksize: int) -> tuple:
    """
    Создает фабрики генераторов для агрегированных данных по странам и годам.
    
    Args:
        chunksize (int): Размер чанка для чтения данных
        
    Returns:
        tuple: Кортеж из двух фабрик-функций:
            - country_generator: фабрика генераторов данных по странам
            - yearly_generator: фабрика генераторов данных по годам
    """
    chunks_list = []
    for chunk in emission_calculation(read_chunks(CSV_FILE, chunksize)):
        chunks_list.append(chunk)
    df = pd.concat(chunks_list, ignore_index=True)
    agg_by_country = df.groupby("Country.Name").agg(list)
    agg_yearly = df.groupby("Year")[["Country.GDP", "Emission.Total"]].sum()
    
    def country_generator():
        """Создает генератор для данных, агрегированных по странам."""
        return (row for _, row in agg_by_country.iterrows())
    
    def yearly_generator():
        """Создает генератор для данных, агрегированных по годам."""
        return (row for _, row in agg_yearly.iterrows())
    
    return country_generator, yearly_generator


def avg_emission_per_capita(agg_data: pd.DataFrame) -> tuple:
    """
    Вычисляет средние выбросы на душу населения для каждой страны.
    
    Args:
        agg_data (pd.DataFrame): Агрегированные данные по странам
        
    Returns:
        tuple: Генератор кортежей (название_страны, средние_выбросы_на_душу_населения)
    """
    return ((country.name, np.mean(country["Emission.Total.Per.Capita"])) for country in agg_data)


def emission_stats(agg_data: pd.DataFrame) -> tuple:
    """
    Вычисляет статистику выбросов: стандартное отклонение и доверительный интервал.
    
    Args:
        agg_data (pd.DataFrame): Агрегированные данные по странам
        
    Returns:
        tuple: Генератор кортежей (название_страны, std_выбросов, доверительный_интервал)
    """
    return ((country.name, np.std(country["Emission.Total"]), 
             1.96 * np.std(country["Emission.Total"])/len(country["Emission.Total"])**0.5) 
             for country in agg_data)


def greenest_and_dirtiest(agg_data: pd.DataFrame) -> tuple:
    """
    Находит 3 самые "зеленые" и 3 самые "грязные" страны по выбросам на душу населения.
    
    Args:
        agg_data (pd.DataFrame): Агрегированные данные по странам
        
    Returns:
        tuple: Кортеж из двух списков:
            - greenest: список из 3 самых "зеленых" стран
            - dirtiest: список из 3 самых "грязных" стран
    """
    avg_per_capita = list(avg_emission_per_capita(agg_data))
    sorted_avg_per_capita = sorted(avg_per_capita, key=lambda item: item[1])
    greenest = sorted_avg_per_capita[:3]
    dirtiest = sorted_avg_per_capita[-3:][::-1]
    return greenest, dirtiest


def highest_and_lowest_emission_std(agg_data: pd.DataFrame) -> tuple:
    """
    Находит страны с наибольшим и наименьшим разбросом выбросов.
    
    Args:
        agg_data (pd.DataFrame): Агрегированные данные по странам
        
    Returns:
        tuple: Кортеж из двух списков:
            - lowest_std: 3 страны с наименьшим стандартным отклонением
            - highest_std: 3 страны с наибольшим стандартным отклонением
    """
    data = list(emission_stats(agg_data))
    sorted_data = sorted(data, key=lambda item: item[1])
    lowest_std = sorted_data[:3]
    highest_std = sorted_data[-3:][::-1]
    return lowest_std, highest_std


def yearly_gdp_and_emission(agg_data: pd.DataFrame) -> pd.DataFrame:
    """
    Анализирует динамику ВВП и выбросов по годам с вычислением скользящего среднего.
    
    Args:
        agg_data (pd.DataFrame): Агрегированные данные по годам
        
    Returns:
        pd.DataFrame: DataFrame с исходными данными и скользящими средними
    """
    yearly = pd.DataFrame(agg_data)
    yearly_with_MA = yearly.join(yearly.rolling(3).mean().rename(
        columns={"Country.GDP": "Country.GDP_MA", "Emission.Total": "Emission_MA.Total"}
    ))
    yearly_with_MA = yearly_with_MA.dropna()
    return yearly_with_MA


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
    country_factory, yearly_factory = country_and_yearly_stats_factory(chunksize)

    country_gen1 = country_factory()
    country_gen2 = country_factory()
    yearly_gen = yearly_factory()

    greenest, dirtiest = greenest_and_dirtiest(country_gen1)
    least_std, most_std = highest_and_lowest_emission_std(country_gen2)
    gdp_and_emission = yearly_gdp_and_emission(yearly_gen)

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    # График 1: Самые "зеленые" страны
    axs[0, 0].bar([c for c, _ in greenest], [v for _, v in greenest], color='green')
    axs[0, 0].set_title('3 самые «зелёные» страны')
    axs[0, 0].set_ylabel('Выбросы на душу населения (кг/чел)')

    # График 2: Самые "грязные" страны
    axs[0, 1].bar([c for c, _ in dirtiest], [v for _, v in dirtiest], color='red')
    axs[0, 1].set_title('3 самые «грязные» страны')
    axs[0, 1].set_ylabel('Выбросы на душу населения (кг/чел)')

    # График 3: Страны с наименьшим разбросом выбросов
    axs[1, 0].bar(
        [c[0] for c in least_std],
        [v[1] for v in least_std],
        yerr=[v[2] for v in least_std],
        capsize=5, color='lightblue'
    )
    axs[1, 0].set_title('Наименьший разброс')
    axs[1, 0].set_ylabel('Стандартное отклонение')

    # График 4: Страны с наибольшим разбросом выбросов
    axs[1, 1].bar(
        [c[0] for c in most_std],
        [v[1] for v in most_std],
        yerr=[v[2] for v in most_std],
        capsize=5, color='lightblue'
    )
    axs[1, 1].set_title('Наибольший разброс')
    axs[1, 1].set_ylabel('Стандартное отклонение')

    # График 5: Динамика ВВП
    axs[0, 2].set_title('Динамика ВВП')
    axs[0, 2].plot(gdp_and_emission.index, gdp_and_emission["Country.GDP"], 
                   color='blue', label='ВВП')
    axs[0, 2].plot(gdp_and_emission.index, gdp_and_emission["Country.GDP_MA"], 
                   color='blue', linestyle='--', label='ВВП (скользящее среднее)')
    axs[0, 2].set_ylabel('ВВП')
    axs[0, 2].legend(loc='upper left')

    # График 6: Динамика выбросов
    axs[1, 2].set_title('Динамика количества выбросов')
    axs[1, 2].plot(gdp_and_emission.index, gdp_and_emission["Emission.Total"], 
                   color='red', label='Выбросы')
    axs[1, 2].plot(gdp_and_emission.index, gdp_and_emission["Emission_MA.Total"], 
                   color='red', linestyle='--', label='Выбросы (скользящее среднее)')
    axs[1, 2].set_ylabel('Выбросы')
    axs[1, 2].legend(loc='upper left')

    plt.show()


if __name__ == "__main__":
    main(int(sys.argv[1]))