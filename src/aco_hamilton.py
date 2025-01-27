import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
import time
import multiprocessing
import matplotlib.pyplot as plt
from functools import partial
from read_cities import set_cities_df, split_into_clusters_kmeans


def calculate_distance_matrix(cities):
    """
    Calculate a distance matrix from a DataFrame of cities.
    """
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i, j] = np.sqrt(
                    (cities.loc[i, "X"] - cities.loc[j, "X"]) ** 2
                    + (cities.loc[i, "Y"] - cities.loc[j, "Y"]) ** 2
                )
    return distance_matrix


def fitness_function(individual, distance_matrix):
    total_distance = 0
    for i in range(len(individual) - 1):
        a, b = individual[i], individual[i + 1]
        total_distance += distance_matrix[a, b]
    return (total_distance,)


def pheromone_update(population, pheromone_matrix, evaporation_rate):
    pheromone_matrix *= 1 - evaporation_rate
    for ind in population:
        total_distance = ind.fitness.values[0]  # reuse stored fitness
        for i in range(len(ind) - 1):
            a, b = ind[i], ind[i + 1]
            pheromone_matrix[a, b] += 1 / total_distance


def biased_selection(individual, pheromone_matrix, distance_matrix, alpha, beta):
    num_cities = len(individual)
    path = [individual[0]]
    while len(path) < num_cities:
        current_city = path[-1]
        unvisited = [city for city in range(num_cities) if city not in path]
        probabilities = [
            (pheromone_matrix[current_city, city] ** alpha)
            * ((1 / distance_matrix[current_city, city]) ** beta)
            for city in unvisited
        ]
        probabilities = np.array(probabilities)
        probabilities /= np.sum(probabilities)
        next_city = random.choices(unvisited, weights=probabilities, k=1)[0]
        path.append(next_city)
    return path


def aco_find_hamilton_path(
    cities, n_ants, n_generations, evaporation_rate, alpha, beta
):
    original_indices = cities.index.tolist()
    cities = cities.reset_index(drop=True)

    distance_matrix = calculate_distance_matrix(cities)

    num_cities = len(cities)
    pheromone_matrix = np.ones((num_cities, num_cities))

    # DEAP setup
    creator.create(
        "FitnessMin", base.Fitness, weights=(-1.0,)
    )  # Minimize total distance
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_cities), num_cities)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.indices
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate", partial(fitness_function, distance_matrix=distance_matrix)
    )
    toolbox.register(
        "pheromone_update",
        partial(
            pheromone_update,
            pheromone_matrix=pheromone_matrix,
            evaporation_rate=evaporation_rate,
        ),
    )
    toolbox.register(
        "select_biased",
        partial(
            biased_selection, distance_matrix=distance_matrix, alpha=alpha, beta=beta
        ),
    )

    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    population = toolbox.population(n=n_ants)
    halloffame = tools.HallOfFame(1)  # Store the best solution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    min_distances = []

    population = toolbox.population(n=n_ants)
    for gen in range(n_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population[:] = toolbox.select(offspring, k=len(population))
        toolbox.pheromone_update(population)
        halloffame.update(population)

        record = stats.compile(population)
        min_distances.append(record["min"])

    best_solution = halloffame[0]
    hamiltonian_path = [original_indices[node] for node in best_solution]

    return hamiltonian_path, stats, min_distances


if __name__ == "__main__":
    n_ants = 200
    n_generations = 1000
    evaporation_rate = 0.5
    alpha = 1
    beta = 2
    all_cities = set_cities_df("data/cities.csv")
    groups = split_into_clusters_kmeans(all_cities, 600)
    cities_df = groups[10]

    start = time.time()
    path, stats, min_distances = aco_find_hamilton_path(
        cities_df, n_ants, n_generations, evaporation_rate, alpha, beta
    )
    end = time.time()

    print("Hamiltonian Path:", path)
    print("Time elapsed:", end - start, "seconds")

    plt.plot(range(n_generations), min_distances, marker="o", linestyle="-", color="b")
    plt.xlabel("Generation")
    plt.ylabel("Minimum Distance")
    plt.title("Evolution of Total Distance Over Generations")
    plt.grid()
    plt.show()
