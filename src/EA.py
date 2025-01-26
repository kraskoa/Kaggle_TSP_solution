from calculate_distance import calculate_path_score, calculate_distance
from read_cities import (
    set_cities_df,
    split_into_clusters_kmeans,
    get_clusters_centroids,
)

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Tworzymy problem minimalizacji (chcemy jak najkrótszą trasę)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
NUM_CITIES = 600
toolbox.register("indices", lambda: random.sample(range(NUM_CITIES), NUM_CITIES))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Lepszy crossover i mutacja
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

# Selekcja
toolbox.register("select", tools.selTournament, tournsize=3)

# Inicjalizacja miast
cities_df = set_cities_df("../data/cities.csv")


# Funkcja oceny
def evaluate(individual):
    return (calculate_path_score(individual, cities_df),)


toolbox.register("evaluate", evaluate)


def main():
    # Tworzymy populację
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)  # Najlepsze rozwiązanie
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Algorytm ewolucyjny
    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=100,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    return pop, stats, hof


if __name__ == "__main__":
    pop, stats, hof = main()

    # Wyświetlanie wyników
    best_ind = hof[0]
    print(
        f"Najlepsza trasa: {best_ind}\nDługość trasy: {best_ind.fitness.values[0]:.2f}"
    )
