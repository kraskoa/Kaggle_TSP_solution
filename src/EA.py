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
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(1, 197770), 600)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register(
    "evaluate", calculate_path_score, cities_df=set_cities_df("../data/cities.csv")
)


def main():
    cities_df = set_cities_df("../data/cities.csv")
    groups = split_into_clusters_kmeans(cities_df, 600)
    centroids = get_clusters_centroids(cities_df)
    centroids = centroids.to_numpy()
    centroids = centroids.tolist()

    # Inicjalizacja populacji
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Algorytm ewolucyjny
    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=40,
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
        "Best individual is: %s\nwith fitness: %s" % (best_ind, best_ind.fitness.values)
    )
