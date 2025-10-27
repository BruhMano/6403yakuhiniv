"""
main.py

Автор: Якухин Иван
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = 'global_emissions.csv'

def read_chunks(filename, chunksize=10000):
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        yield chunk

def emission_calculation(chunks):
    gases = ['Emissions.Production.CO2.Total', 'Emissions.Production.CH4', 'Emissions.Production.N2O']
    columns = ["Year", "Country.Name", "Country.GDP", "Emission.Total", "Emission.Total.Per.Capita"]
    for chunk in chunks:
        chunk['Emission.Total'] = chunk[gases].sum(axis = 1)
        chunk['Emission.Total.Per.Capita'] = chunk['Emission.Total'] / chunk['Country.Population'] * 1e6
        yield chunk[columns]

def country_and_yearly_stats_factory(chunksize):
    chunks_list = []
    for chunk in emission_calculation(read_chunks(CSV_FILE, chunksize)):
        chunks_list.append(chunk)
    df = pd.concat(chunks_list, ignore_index=True)
    agg_by_country = df.groupby("Country.Name").agg(list)
    agg_yearly = df.groupby("Year")[["Country.GDP", "Emission.Total"]].sum()
    
    # Создаем фабрики генераторов вместо самих генераторов
    def country_generator():
        return (row for _, row in agg_by_country.iterrows())
    
    def yearly_generator():
        return (row for _, row in agg_yearly.iterrows())
    
    return country_generator, yearly_generator

def avg_emission_per_capita(agg_data):
    return ((counntry.name, np.mean(counntry["Emission.Total.Per.Capita"])) for counntry in agg_data)

def emission_stats(agg_data):
    return ((country.name, np.std(country["Emission.Total"]), 
             1.96 * np.std(country["Emission.Total"])/len(country["Emission.Total"])**0.5) 
             for country in agg_data)

def greenest_and_dirtiest(agg_data):
    avg_per_capita = list(avg_emission_per_capita(agg_data))
    sorted_avg_per_capita = sorted(avg_per_capita, key = lambda item: item[1])
    greenest = sorted_avg_per_capita[:3]
    dirtiest = sorted_avg_per_capita[-3:][::-1]
    return greenest, dirtiest

def highest_and_lowest_emission_std(agg_data):
    data = list(emission_stats(agg_data))
    sorted_data = sorted(data, key = lambda item: item[1])
    lowest_std = sorted_data[:3]
    highest_std = sorted_data[-3:][::-1]
    return lowest_std, highest_std

def yearly_gdp_and_emission(agg_data):
    yearly = pd.DataFrame(agg_data)
    yearly_with_MA = yearly.join(yearly.rolling(3).mean().rename(columns={"Country.GDP":"Country.GDP_MA","Emission.Total":"Emission_MA.Total"}))
    yearly_with_MA = yearly_with_MA.dropna()
    return yearly_with_MA

def main():
    country_factory, yearly_factory = country_and_yearly_stats_factory(1000)

    country_gen1 = country_factory()
    country_gen2 = country_factory()
    yearly_gen = yearly_factory()

    greenest, dirtiest = greenest_and_dirtiest(country_gen1)
    least_std, most_std = highest_and_lowest_emission_std(country_gen2)
    gdp_and_emission = yearly_gdp_and_emission(yearly_gen)

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(wspace=0.5,hspace=0.5)

    axs[0, 0].bar([c for c, _ in greenest], [v for _, v in greenest], color='green')
    axs[0, 0].set_title('3 самые «зелёные» страны')
    axs[0, 0].set_ylabel('Выбросы на душу населения (кг/чел)')

    axs[0, 1].bar([c for c, _ in dirtiest], [v for _, v in dirtiest], color='red')
    axs[0, 1].set_title('3 самые «грязные» страны')
    axs[0, 1].set_ylabel('Выбросы на душу населения (кг/чел)')

    # # Задание 2
    axs[1, 0].bar(
        [c[0] for c in least_std],
        [v[1] for v in least_std],
        yerr=[v[2] for v in least_std],
        capsize=5, color='lightblue'
    )
    axs[1, 0].set_title('Наименьший разброс')
    axs[1, 0].set_ylabel('Стандартное отклонение')

    axs[1, 1].bar(
        [c[0] for c in most_std],
        [v[1] for v in most_std],
        yerr=[v[2] for v in most_std],
        capsize=5, color='lightblue'
    )
    axs[1, 1].set_title('Наибольший разброс')
    axs[1, 1].set_ylabel('Стандартное отклонение')

    axs[0, 2].set_title('Динамика ВВП')
    axs[0, 2].plot(gdp_and_emission.index, gdp_and_emission["Country.GDP"], color='blue', label='ВВП')
    axs[0, 2].plot(gdp_and_emission.index, gdp_and_emission["Country.GDP_MA"], color='blue', linestyle = '--', label='ВВП (скользящее среднее)')
    axs[0, 2].set_ylabel('ВВП')
    axs[0, 2].legend(loc='upper left')

    axs[1, 2].set_title('Динамика количества выбросов')
    axs[1, 2].plot(gdp_and_emission.index, gdp_and_emission["Emission.Total"], color='red', label='Выбросы')
    axs[1, 2].plot(gdp_and_emission.index, gdp_and_emission["Emission_MA.Total"], color='red', linestyle = '--', label='Выбросы (скользящее среднее)')
    axs[1, 2].set_ylabel('Выбросы')
    axs[1, 2].legend(loc='upper left')

    plt.show()


if __name__ == "__main__":
    main()
