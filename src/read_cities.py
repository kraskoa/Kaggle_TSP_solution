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


def get_clusters_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the centroids of the clusters.

    Parameters:
    - df (pd.DataFrame): DataFrame with city data.

    Returns:
    - centroids (pd.DataFrame): DataFrame with the centroids of the clusters.
    """
    north_pole_cluster_number = df.loc[0, "Cluster"]
    centroids = df.groupby("Cluster")[["X", "Y"]].mean()
    centroids["IsNorthPole"] = 0
    centroids.loc[north_pole_cluster_number, "IsNorthPole"] = 1
    return centroids


if __name__ == "__main__":

    cities_df = set_cities_df("../data/cities.csv")
    print(cities_df.head())

    groups = split_into_clusters_kmeans(cities_df, 600)
    print(
        f"klaster 0:{len(groups[0])}, klaster 1:{len(groups[1])}, klaster 2:{len(groups[2])}, klaster 3:{len(groups[3])}, klaster 4:{len(groups[4])}, klaster 5:{len(groups[5])}, klaster 6:{len(groups[6])}, klaster 7:{len(groups[7])}, klaster 8:{len(groups[8])}, klaster 9:{len(groups[9])}"
    )
    # print(groups[1].head())
    # print(groups[2].head())
    # print(groups[3].head())
    max_length_of_group = 0

    # for i in range(399):
    #     print(f"Grupa {i} - liczba rekordów: {len(groups[i])}")
    #     print(groups[i].head(), "\n")
    #     max_length_of_group = max(max_length_of_group, len(groups[i]))

    print(max_length_of_group)
    print(cities_df.head())

    centroids = get_clusters_centroids(cities_df)
    print(centroids.head(78))
    # print(centroids.loc[77])
    print(len(centroids))
