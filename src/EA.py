from calculate_distance import (
    calculate_path_score,
    calculate_distance,
    calculate_centroids_path,
)
from read_cities import (
    set_cities_df,
    get_clusters_centroids,
    split_into_clusters_kmeans,
)
import numpy as np
import random
from deap import base, creator, tools, algorithms

# Inicjalizacja miast
cities_df = set_cities_df("../data/cities.csv")
split_into_clusters_kmeans(cities_df, 600)

# Inicjalizacja centroidów
centroid_df = get_clusters_centroids(cities_df)

# Sprawdzenie, czy mamy poprawny centroid
if not centroid_df[centroid_df["IsNorthPole"] == 1].empty:
    CENTROID_INDEX = centroid_df.loc[centroid_df["IsNorthPole"] == 1].index[0]
else:
    raise ValueError("Brak centroidu z IsNorthPole == 1!")

# Tworzymy problem minimalizacji (chcemy jak najkrótszą trasę)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
NUM_CENTROIDS = 600


def generate_individual():
    """Tworzy losową trasę zaczynającą i kończącą się na centroidzie"""
    centroids_path = list(range(NUM_CENTROIDS))
    centroids_path.remove(CENTROID_INDEX)
    random.shuffle(centroids_path)
    print(len(centroids_path))
    print(CENTROID_INDEX)
    return [CENTROID_INDEX] + centroids_path + [CENTROID_INDEX]


toolbox.register("indices", generate_individual)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def crossover_with_fixed_start_end(ind1, ind2):
    """Krzyżowanie Ordered Crossover (OX) z zachowaniem pierwszego i ostatniego miasta"""
    start, end = ind1[0], ind1[-1]

    temp1, temp2 = ind1[1:-1], ind2[1:-1]

    # Sprawdzenie poprawności listy przed krzyżowaniem
    if len(temp1) < 2 or len(temp2) < 2:
        return (
            ind1,
            ind2,
        )  # Jeśli długość permutacji jest zbyt mała, zwracamy oryginalne osobniki

    # Utworzenie listy unikalnych klastrów
    cluster_indices = list(set(temp1 + temp2))
    if len(cluster_indices) != NUM_CENTROIDS - 1:
        raise ValueError("Duplicate or missing cluster indices during crossover")

    # Tworzenie mapowania klastrów do nowego zakresu
    mapping = {cluster: i for i, cluster in enumerate(cluster_indices)}
    reverse_mapping = {i: cluster for cluster, i in mapping.items()}

    # Przekształcenie genów na nowy zakres
    temp1_mapped = [mapping[gene] for gene in temp1]
    temp2_mapped = [mapping[gene] for gene in temp2]

    # Wykonanie krzyżowania na zamapowanych genach
    tools.cxOrdered(temp1_mapped, temp2_mapped)

    # Przekształcenie genów z powrotem na oryginalne identyfikatory klastrów
    temp1_new = [reverse_mapping[gene] for gene in temp1_mapped]
    temp2_new = [reverse_mapping[gene] for gene in temp2_mapped]

    # Aktualizacja osobników z zachowaniem pierwszego i ostatniego elementu
    ind1[:] = [start] + temp1_new + [end]
    ind2[:] = [start] + temp2_new + [end]

    return ind1, ind2


# Rejestracja nowego operatora krzyżowania
toolbox.register("mate", crossover_with_fixed_start_end)


def swap_mutation_with_fixed_points(individual):
    """Losowa zamiana dwóch miast z zachowaniem pierwszego i ostatniego miasta"""
    idx1, idx2 = random.sample(range(1, len(individual) - 1), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return (individual,)


toolbox.register("mutate", swap_mutation_with_fixed_points)

# Selekcja
toolbox.register("select", tools.selTournament, tournsize=3)


def evaluate(individual):
    return (calculate_centroids_path(individual, centroid_df),)


toolbox.register("evaluate", evaluate)


def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Sprawdzenie populacji przed startem algorytmu
    for ind in pop:
        assert (
            len(ind) == NUM_CENTROIDS + 1
        ), f"Błąd: osobnik ma {len(ind)} miast, oczekiwano {NUM_CENTROIDS + 1}"
        assert (
            ind[0] == CENTROID_INDEX and ind[-1] == CENTROID_INDEX
        ), f"Błąd: trasa nie zaczyna/kończy się na {CENTROID_INDEX}"

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=1000,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    return pop, stats, hof


if __name__ == "__main__":
    pop, stats, hof = main()
    best_ind = hof[0]
    best_fitness = best_ind.fitness.values[0]  # Uzyskaj wartość fitness przed konwersją
    best_ind = [int(x) for x in best_ind]  # Konwersja elementów na typ int
    print(f"Najlepsza trasa: {best_ind}\nDługość trasy: {best_fitness:.2f}")
