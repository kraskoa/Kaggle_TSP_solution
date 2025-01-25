import pandas as pd
from sklearn.cluster import KMeans
import math
from sympy import isprime


def set_cities_df(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV file with city data and returns a DataFrame.

    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - df (pd.DataFrame): DataFrame with city data.
    """
    df = pd.read_csv(filepath, index_col="CityId")
    df["IsPrime"] = df.index.map(isprime)
    return df


def split_into_clusters_kmeans(df: pd.DataFrame, n_clusters: int) -> list:
    """
    Splits cities into groups using the K-Means algorithm.

    Parameters:
    - df (pd.DataFrame): DataFrame with city data.
    - n_clusters (int): Number of clusters.

    Returns:
    - groups (list): List containing DataFrames with cities in each group.
    """
    X = df[["X", "Y"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)
    groups = [group for _, group in df.groupby("Cluster")]
    return groups


if __name__ == "__main__":

    cities_df = set_cities_df("../data/cities.csv")
    print(cities_df.head())

    groups = split_into_clusters_kmeans(cities_df, 350)
    print(groups[0].head())
    print(groups[1].head())
    print(groups[2].head())
    print(groups[3].head())

    for i in range(10):
        print(f"Grupa {i} - liczba rekordów: {len(groups[i])}")
        print(groups[i].head(), "\n")
