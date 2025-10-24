"""
main.py

Автор: Якухин Иван
"""

import pandas as pd
import numpy as np

CSV_FILE = 'global_emissions.csv'
PARQUET_FILE = 'global_emissions.parquet'

def read_chunks(filename, chunksize=10000):
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        yield chunk

def total_emission(chunks):
    gases = ['Emissions.Production.CO2.Total', 'Emissions.Production.CH4', 'Emissions.Production.N2O']
    for chunk in chunks:
        aggregaterd = chunk.groupby('Country.Name')[gases].mean().sum(axis = 1)
        aggregaterd.name = 'Emissions.Production.Total'
        aggregaterd = aggregaterd.reset_index()
        chunk = pd.merge(chunk, aggregaterd)
        population = chunk.groupby('Country.Name')
    return chunk

def main() -> None:
    df_gen = read_chunks(CSV_FILE, 100)
    total_emission(df_gen).to_csv("test.csv")


if __name__ == "__main__":
    main()
