import numpy as np
import pandas as pd
from calculate_distance import (
    calculate_path_score,
    calculate_distance,
    calculate_local_path,
)
from read_cities import (
    set_cities_df,
    get_clusters_centroids,
    split_into_clusters_kmeans,
)
import ast


def main():
    cities_df = set_cities_df("../data/cities.csv")
    split_into_clusters_kmeans(cities_df, 600)
    centroids_df = get_clusters_centroids(cities_df)
    print(centroids_df.head())
    best_routes_per_cluster_df = pd.read_csv(
        "../data/best_routes_per_cluster.csv", index_col="clusterID"
    )
    best_centroids_hamiltonian_cycle_df = pd.read_csv(
        "../data/best_route_centroids.csv"
    )
    print(best_centroids_hamiltonian_cycle_df)
    print(best_routes_per_cluster_df)
    final_city_path = []
    cluster_ids = best_centroids_hamiltonian_cycle_df["ClusterId"].tolist()
    comeback = False
    for cluster_id in cluster_ids:
        ordered_indices_str = best_routes_per_cluster_df.loc[cluster_id].values[0]
        ordered_indices = ast.literal_eval(ordered_indices_str)
        if cluster_id != cluster_ids[0] or not comeback:
            for city_id in ordered_indices:
                final_city_path.append(int(city_id))
        if cluster_id == cluster_ids[-1] and comeback:
            final_city_path.append(0)
        comeback = True

    final_score = calculate_path_score(final_city_path, cities_df)
    print(final_score)

    final_city_path_df = pd.DataFrame(final_city_path, columns=["CityId"])
    final_city_path_df.to_csv("final_city_path.csv", index=False)

    with open("final_score.txt", "w") as file:
        file.write(f"Final Score: {final_score}\n")


if __name__ == "__main__":
    main()
