import numpy as np
import pandas as pd
import read_cities
import math


def calculate_distance(x1, y1, x2, y2, is_prime, is_tenth):
    if not is_tenth:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    elif is_prime and is_tenth:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    else:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * 1.1


def calculate_path_score(path, cities_df):
    score = 0
    for i in range(1, len(path)):
        city1 = cities_df.loc[path[i - 1]]
        city2 = cities_df.loc[path[i]]
        score += calculate_distance(
            city1["X"],
            city1["Y"],
            city2["X"],
            city2["Y"],
            city1["IsPrime"],
            (i - 1) % 10 == 0,
        )
    return score


def calculate_centroids_path(centroids_path, centroids_df):
    coords = centroids_df.loc[centroids_path][["X", "Y"]].values
    diffs = np.diff(coords, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


if __name__ == "__main__":
    cities_df = read_cities.set_cities_df("../data/cities.csv")
    dumbest_path = cities_df.index.values[1:]
    path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dumb_score = calculate_path_score(dumbest_path, cities_df)
    print(dumb_score)
    score = calculate_path_score(path, cities_df)
    print(score)
    # Output: 0.0
    path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    score = calculate_path_score(path, cities_df)
    print(score)
    # Output: 1.1
    path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    score = calculate_path_score(path, cities_df)
    print(score)
    # Output: 2.2
    path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    score = calculate_path_score(path, cities_df)
    print(score)
    # Output: 3.3
    path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    score = calculate_path_score(path, cities_df)
    print(score)
    # Output: 4.4
    path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    score = calculate_path_score(path, cities_df)
    print(score)
    # Output: 5.5
    path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    score = calculate_path_score(path, cities_df)
    print(score)
    # Output: 6.6
