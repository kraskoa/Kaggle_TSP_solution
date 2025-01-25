import numpy as np


def calucalate_distance(x1, y1, x2, y2, is_prime, is_tenth):
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
        score += calucalate_distance(
            city1["X"],
            city1["Y"],
            city2["X"],
            city2["Y"],
            city1["IsPrime"],
            i % 10 == 0,
        )
    return score
