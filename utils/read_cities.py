import pandas as pd


def read_cities():
    cities_df = pd.read_csv("data/cities.csv")
    cities = {row["CityID"]: (row["X"], row["Y"]) for _, row in cities_df.iterrows()}
    return cities
