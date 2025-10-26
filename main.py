"""
main.py

Автор: Якухин Иван
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = 'global_emissions.csv'
PARQUET_FILE = 'global_emissions.parquet'

def read_chunks(filename, chunksize=10000):
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        yield chunk

def total_emission(chunks):
    gases = ['Emissions.Production.CO2.Total', 'Emissions.Production.CH4', 'Emissions.Production.N2O']
    for chunk in chunks:
        chunk['Emission.Total'] = chunk[gases].sum(axis = 1)
        yield chunk

def avg_emission_per_capita(chunks):
    total = {}
    count = {}
    for chunk in chunks:
        chunk['Emission.Total.Per.Capita'] = chunk['Emission.Total'] / chunk['Country.Population']
        for _, row in chunk.iterrows():
            country_name = row['Country.Name']
            per_capita = row['Emission.Total.Per.Capita']
            total[country_name] = total.get(country_name, 0) + per_capita
            count[country_name] = count.get(country_name, 0) + 1
    return ((country, total[country] / count[country] * 1e6) for country in total)

def emission_list_per_country(chunks):
    emission = {}
    for chunk in chunks:
        for _, row in chunk.iterrows():
            country_name = row['Country.Name']
            total = row['Emission.Total']
            if country_name not in emission:
                emission[country_name] = []
            emission[country_name].append(total)
    return ((key, val) for key, val in emission.items())

def yearly_gdp_and_emission(chunks):
    yearly = pd.DataFrame(columns = ["Year", "GDP", "Emission"])
    yearly = yearly.set_index("Year")
    for chunk in chunks:
        for _, row in chunk.iterrows():
            year = row["Year"]
            gdp = row["Country.GDP"]
            emission = row["Emission.Total"]
            if year not in yearly.index:
                yearly.loc[year] = [0,0]
            yearly.loc[year] += [gdp, emission]
    yearly_MA = yearly.join(yearly.rolling(3).mean().rename(columns={"GDP":"GDP_MA","Emission":"Emission_MA"}))
    yearly_MA = yearly_MA.dropna()
    return (new_row for _, new_row in yearly_MA.iterrows())

def greenest_and_dirtiest(chunksize):
    avg_per_capita = dict(avg_emission_per_capita(total_emission(read_chunks(CSV_FILE, chunksize))))
    sorted_avg_per_capita = sorted(avg_per_capita.items(), key = lambda item: item[1])
    greenest = sorted_avg_per_capita[:3]
    dirtiest = sorted_avg_per_capita[-3:][::-1]
    return greenest, dirtiest

def emission_stats(data):
    for country, emission in data:
        if len(emission) < 2:
            continue
        arr = np.array(emission)
        mean = arr.mean()
        std = arr.std()
        ci = 1.96 * std / np.sqrt(len(arr))
        yield country, mean, std, ci

def highest_and_lowest_emission_std(chunksize):
    data = list(emission_stats(emission_list_per_country(total_emission(read_chunks(CSV_FILE, chunksize)))))
    sorted_data = sorted(data, key = lambda item: item[2])
    lowest_std = sorted_data[:3]
    highest_std = sorted_data[-3:][::-1]
    return lowest_std, highest_std

def main():
    greenest, dirtiest = greenest_and_dirtiest(100)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].bar([c for c, _ in greenest], [v for _, v in greenest], color='green')
    axs[0, 0].set_title('3 самые «зелёные» страны')
    axs[0, 0].set_xlabel('Выбросы на душу населения')

    axs[0, 1].bar([c for c, _ in dirtiest], [v for _, v in dirtiest], color='red')
    axs[0, 1].set_title('3 самые «грязные» страны')
    axs[0, 1].set_xlabel('Выбросы на душу населения')

    # Задание 2
    least_var, most_var = highest_and_lowest_emission_std(100)
    axs[1, 0].bar(
        [c[0] for c in least_var],
        [v[2] for v in least_var],
        yerr=[v[3] for v in least_var],
        capsize=5, color='lightblue'
    )
    axs[1, 0].set_title('Наименьший разброс')
    axs[1, 0].set_ylabel('Стандартное отклонение')

    axs[1, 1].bar(
        [c[0] for c in most_var],
        [v[2] for v in most_var],
        yerr=[v[3] for v in most_var],
        capsize=5, color='lightblue'
    )
    axs[1, 1].set_title('Наибольший разброс')
    axs[1, 1].set_ylabel('Стандартное отклонение')

    plt.show()

    df = pd.DataFrame(yearly_gdp_and_emission(total_emission(read_chunks(CSV_FILE, 100))))
    fig, axs = plt.subplots(2, figsize=(10, 6))
    axs[0].set_title('Динамика ВВП')
    axs[0].plot(df.index, df["GDP"], color='blue')
    axs[0].plot(df.index, df["GDP_MA"], color='blue', linestyle = '--')
    axs[0].set_ylabel('GDP')
    axs[1].set_title('Динамика количества выбросов')
    axs[1].plot(df.index, df["Emission"], color='red')
    axs[1].plot(df.index, df["Emission_MA"], color='red', linestyle = '--')
    axs[1].set_ylabel('Emissions')
    plt.show()


if __name__ == "__main__":
    main()
