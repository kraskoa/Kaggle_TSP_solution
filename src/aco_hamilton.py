import numpy as np
import random
from scipy.spatial.distance import cdist
from utils.read_cities import (
    set_cities_df,
    split_into_clusters_kmeans,
    get_clusters_centroids,
)


def aco(cities, n_ants, n_iterations, alpha, beta, rho):
    coordinates = np.array([[c["X"], c["Y"]] for c in cities])
    distance_matrix = cdist(coordinates, coordinates)

    pheromones = np.ones((n_cities, n_cities))

    n_cities = len(cities)
    best_path = None
    best_cost = float("inf")

    def calculate_cost(path):
        total_cost = 0
        speed = 1.0
        for i in range(len(path) - 1):
            total_cost += distance_matrix[path[i], path[i + 1]] / speed
            if (i + 1) % 10 == 0 and not cities[path[i + 1]]["isPrime"]:
                speed *= 0.9
        return total_cost

    for _ in range(n_iterations):
        all_paths = []
        all_costs = []

        for _ in range(n_ants):
            visited = set()
            path = [random.randint(0, n_cities - 1)]
            visited.add(path[0])

            while len(path) < n_cities:
                current = path[-1]
                probabilities = []
                for j in range(n_cities):
                    if j not in visited:
                        pheromone = pheromones[current, j] ** alpha
                        heuristic = (1 / distance_matrix[current, j]) ** beta
                        probabilities.append(pheromone * heuristic)
                    else:
                        probabilities.append(0)
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                next_city = np.random.choice(range(n_cities), p=probabilities)
                path.append(next_city)
                visited.add(next_city)

            path.append(path[0])
            cost = calculate_cost(path)
            all_paths.append(path)
            all_costs.append(cost)

        for i in range(len(all_costs)):
            if all_costs[i] < best_cost:
                best_cost = all_costs[i]
                best_path = all_paths[i]

        pheromones *= 1 - rho
        for i, path in enumerate(all_paths):
            for j in range(len(path) - 1):
                pheromones[path[j], path[j + 1]] += 1 / all_costs[i]

    return best_path, best_cost


if __name__ == "__main__":
    n_ants = 20
    n_iterations = 100
    alpha = 1  # wpływ feromonów
    beta = 2  # wpływ heurystyki (1 / odległość)
    rho = 0.5  # współczynnik parowania feromonów

    all_cities = set_cities_df("data/cities.csv")
    groups = split_into_clusters_kmeans(all_cities, 600)
    cities = groups[0]

    best_path, best_cost = aco(cities, n_ants, n_iterations, alpha, beta, rho)
    print("Najlepsza ścieżka:", best_path)
    print("Koszt:", best_cost)
