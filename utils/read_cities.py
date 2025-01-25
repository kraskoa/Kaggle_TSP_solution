import pandas as pd
from sklearn.cluster import KMeans
import math
from sympy import isprime


def set_cities_df(filepath: str) -> pd.DataFrame:
    """
    Wczytuje plik CSV z danymi miast i zwraca DataFrame.

    Parametry:
    - filepath (str): Ścieżka do pliku CSV.

    Zwraca:
    - df (pd.DataFrame): DataFrame z danymi miast.
    """
    df = pd.read_csv(filepath, index_col="CityId")
    df["IsPrime"] = df.index.map(isprime)
    return df


def podziel_na_grupy_kmeans(plik_csv, rekordy_na_czesc=200):
    """
    Dzieli duży plik CSV na mniejsze grupy zawierające określoną liczbę rekordów,
    przy użyciu algorytmu K-Means do grupowania punktów blisko siebie.

    Parametry:
    - plik_csv (str): Ścieżka do pliku CSV zawierającego kolumny 'CityId', 'X', 'Y'.
    - rekordy_na_czesc (int): Liczba rekordów w każdej grupie.

    Zwraca:
    - list_of_groups (list of pd.DataFrame): Lista DataFrame'ów, każdy zawierający grupę punktów.
    """
    # Wczytaj dane z pliku CSV
    df = pd.read_csv(plik_csv, index_col="CityId")

    # Oblicz liczbę części
    liczba_czesci = math.ceil(len(df) / rekordy_na_czesc)

    # Inicjalizuj algorytm K-Means
    kmeans = KMeans(n_clusters=liczba_czesci, random_state=42)

    # Dopasuj model do danych
    df["cluster"] = kmeans.fit_predict(df[["X", "Y"]])

    # Grupuj dane według klastra
    grupy = df.groupby("cluster")

    # Zbiór do przechowywania grup DataFrame
    list_of_groups = []

    # Iteruj przez grupy i dodawaj je do listy
    for cluster_id, grupa in grupy:
        if not grupa.empty:
            # Usuń kolumnę pomocniczą
            grupa_clean = grupa.drop(["cluster"], axis=1)
            list_of_groups.append(grupa_clean)

    return list_of_groups


# Przykładowe użycie
if __name__ == "__main__":
    # plik_csv = "../data/cities.csv"
    # rekordy_na_czesc = 200

    # grupy_kmeans = podziel_na_grupy_kmeans(plik_csv, rekordy_na_czesc)

    # # Wyświetlenie informacji o pierwszych kilku grupach
    # for idx, grupa in enumerate(grupy_kmeans[:5], start=1):
    #     print(f"Grupa K-Means {idx} - liczba rekordów: {len(grupa)}")
    #     print(grupa.head(), "\n")

    # print(f"Łączna liczba grup (K-Means): {len(grupy_kmeans)}")

    cities_df = set_cities_df("../data/cities.csv")
    print(cities_df.head())
