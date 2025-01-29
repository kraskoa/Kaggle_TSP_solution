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
import numpy as np
import random
from deap import base, creator, tools, algorithms
import multiprocessing
import matplotlib.pyplot as plt
import time
from functools import partial
import itertools

# KONFIGURACJA ALGORYTMU
POP_SIZE = 2000  # Rozmiar populacji
NGEN = 100  # Liczba generacji
CXPB = 0.7  # Prawdopodobieństwo krzyżowania
MUTPB = 0.2  # Prawdopodobieństwo mutacji

cities_df = set_cities_df("../data/cities.csv")
split_into_clusters_kmeans(cities_df, 600)

centroid_df = get_clusters_centroids(cities_df)

if not centroid_df[centroid_df["IsNorthPole"] == 1].empty:
    CENTROID_INDEX = centroid_df.loc[centroid_df["IsNorthPole"] == 1].index[0]
else:
    raise ValueError("Brak centroidu z IsNorthPole == 1!")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
NUM_CENTROIDS = 600


def generate_individual():
    centroids_path = list(range(NUM_CENTROIDS))
    centroids_path.remove(CENTROID_INDEX)
    random.shuffle(centroids_path)
    return [CENTROID_INDEX] + centroids_path + [CENTROID_INDEX]


toolbox.register("indices", generate_individual)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def crossover_with_fixed_start_end(ind1, ind2):
    start, end = ind1[0], ind1[-1]

    temp1, temp2 = ind1[1:-1], ind2[1:-1]

    if len(temp1) < 2 or len(temp2) < 2:
        return (
            ind1,
            ind2,
        )

    cluster_indices = list(set(temp1 + temp2))
    if len(cluster_indices) != NUM_CENTROIDS - 1:
        raise ValueError("Duplicate or missing cluster indices during crossover")

    mapping = {cluster: i for i, cluster in enumerate(cluster_indices)}
    reverse_mapping = {i: cluster for cluster, i in mapping.items()}

    temp1_mapped = [mapping[gene] for gene in temp1]
    temp2_mapped = [mapping[gene] for gene in temp2]

    tools.cxOrdered(temp1_mapped, temp2_mapped)

    temp1_new = [reverse_mapping[gene] for gene in temp1_mapped]
    temp2_new = [reverse_mapping[gene] for gene in temp2_mapped]

    ind1[:] = [start] + temp1_new + [end]
    ind2[:] = [start] + temp2_new + [end]

    return ind1, ind2


toolbox.register("mate", crossover_with_fixed_start_end)


def swap_mutation(individual):
    idx1, idx2 = random.sample(range(1, len(individual) - 1), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return (individual,)


toolbox.register("mutate", swap_mutation)


# def mutate_2opt(individual):
#     """Mutacja 2-Opt."""
#     start, end = individual[0], individual[-1]
#     temp = individual[1:-1]

#     # Jeśli permutacja jest za krótka, nie ma co optymalizować
#     if len(temp) < 2:
#         return (individual,)

#     i, j = sorted(random.sample(range(len(temp)), 2))
#     # Odwróć podsegment
#     temp[i:j] = reversed(temp[i:j])

#     individual[:] = [start] + temp + [end]
#     return (individual,)


# toolbox.register("mutate", mutate_2opt)


# def mutate_3opt(individual, centroid_df):
#     """
#     Mutacja 3-Opt z zachowaniem pierwszego i ostatniego miasta,
#     z WYBOREM najlepszej opcji (najkrótsza trasa).

#     - Wycinamy trzy krawędzie w "temp" (środkowej części).
#     - Generujemy wszystkie możliwe warianty 3-Opt (odwrócenia segmentów B i C
#       oraz ewentualną zamianę kolejności B <-> C).
#     - Liczymy koszt i wybieramy najlepszy (najmniejszy).
#     """

#     start, end = individual[0], individual[-1]
#     temp = individual[1:-1]

#     if len(temp) < 3:
#         return (individual,)

#     i, j, k = sorted(random.sample(range(len(temp)), 3))

#     A = temp[:i]
#     B = temp[i:j]
#     C = temp[j:k]
#     D = temp[k:]

#     def rev_if_needed(segment, do_reverse):
#         return list(reversed(segment)) if do_reverse else segment

#     transformations = set()

#     for swap, revB, revC in itertools.product(
#         [False, True], [False, True], [False, True]
#     ):
#         B_ = rev_if_needed(B, revB)
#         C_ = rev_if_needed(C, revC)

#         if not swap:
#             new_temp = A + B_ + C_ + D
#         else:
#             new_temp = A + C_ + B_ + D

#         transformations.add(tuple(new_temp))

#     orig_path = [start] + temp + [end]
#     best_cost = calculate_local_path(orig_path, centroid_df)
#     best_temp = temp

#     for t in transformations:
#         candidate = [start] + list(t) + [end]
#         cost_candidate = calculate_local_path(candidate, centroid_df)
#         if cost_candidate < best_cost:
#             best_cost = cost_candidate
#             best_temp = list(t)

#     individual[:] = [start] + best_temp + [end]
#     return (individual,)


# toolbox.register("mutate", partial(mutate_3opt, centroid_df=centroid_df))

toolbox.register("select", tools.selTournament, tournsize=3)


def evaluate(individual):
    """Ocena osobnika – obliczenie długości lokalnej ścieżki."""
    return (calculate_local_path(individual, centroid_df),)


toolbox.register("evaluate", evaluate)


def main():
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=POP_SIZE)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    pool.close()
    pool.join()

    return pop, stats, hof, log


if __name__ == "__main__":
    start_time = time.time()
    pop, stats, hof, log = main()
    print(f"Czas wykonania: {time.time() - start_time:.2f} s")

    best_ind = hof[0]
    best_fitness = best_ind.fitness.values[0]
    best_ind = [int(x) for x in best_ind]
    print(f"Najlepsza trasa: {best_ind}\nDługość trasy: {best_fitness:.2f}")

    try:
        with open("best_route.txt", "r") as f:
            lines = f.readlines()
            saved_fitness = float(lines[1].split(": ")[1])
    except (FileNotFoundError, IndexError, ValueError):
        saved_fitness = float("inf")

    if best_fitness < saved_fitness:
        with open("best_route.txt", "w") as f:
            f.write(f"Najlepsza trasa: {best_ind}\nDługość trasy: {best_fitness:.2f}")

    # WYKRESY Z LOGÓW
    gen = log.select("gen")
    min_ = log.select("min")
    avg = log.select("avg")
    max_ = log.select("max")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(gen, avg, "g--", label="Średnia")
    ax1.set_xlabel("Generacja")
    ax1.set_ylabel("Średnia wartość funkcji oceny")
    ax1.set_title("Średnia wartość funkcji oceny w każdej generacji")
    ax1.legend(loc="best")
    ax1.grid(True)

    ax2.plot(gen, min_, "ro-", label="Minimalna", markersize=6)
    ax2.set_xlabel("Generacja")
    ax2.set_ylabel("Minimalna wartość funkcji oceny")
    ax2.set_title("Minimalna wartość funkcji oceny w każdej generacji")
    ax2.legend(loc="best")
    ax2.grid(True)

    fig.suptitle(
        "Parametry algorytmu ewolucyjnego:\n"
        f"Populacja: {POP_SIZE}, "
        f"Generacje: {NGEN}, "
        f"p_krzyżowania: {CXPB}, "
        f"p_mutacji: {MUTPB}",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("fitness_plots.png")
    plt.show()
