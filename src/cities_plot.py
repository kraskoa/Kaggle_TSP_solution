import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import read_cities

# Wczytanie danych
df_cities = read_cities.set_cities_df("../data/cities.csv")

# Resetowanie indeksu, aby CityId stał się kolumną
df_cities = df_cities.reset_index()

fig = plt.figure(figsize=(10, 10))
plt.scatter(
    df_cities["X"],
    df_cities["Y"],
    marker=".",
    c=(df_cities["CityId"] != 0).astype(int),
    cmap="Set1",
    alpha=0.6,
    s=500 * (df_cities["CityId"] == 0).astype(int) + 1,
)
plt.show()
