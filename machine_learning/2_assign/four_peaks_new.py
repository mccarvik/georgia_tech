"""
Four peaks problem
"""

import pdb
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np


LENGTH = 50
THRESHOLD = 20

def generate_candidate_solution():
    return [random.choice([0, 1]) for _ in range(LENGTH)]


def evaluate_solution(solution):
    num_zeros = 0
    num_ones = 0
    for bit in solution:
        if bit == 0:
            num_zeros += 1
        else:
            break
    for bit in reversed(solution):
        if bit == 1:
            num_ones += 1
        else:
            break
    if num_zeros > THRESHOLD and num_ones > THRESHOLD:
        return num_zeros + num_ones + 100
    else:
        return num_zeros + num_ones


def generate_neighbor_solution(solution):
    """
    Generate a neighbor solution by flipping a random bit
    """
    index = random.randint(0, LENGTH - 1)
    neighbor = solution[:]
    if neighbor[index] == 0:
        neighbor[index] = 1
    else:
        neighbor[index] = 0
    return neighbor


def hill_climbing(runs=3000, max_no_improve=3, random_starts=1):
    """
    Hill climbing algorithm
    """
    best_solution = 0
    best_score = 0

    saved_scores = []
    start_time = time.perf_counter()
    for _ in range(random_starts):
        current_solution = generate_candidate_solution()
        current_score = evaluate_solution(current_solution)

        no_improve = 0
        for x in range(runs):
            neighbor = generate_neighbor_solution(current_solution)
            neighbor_score = evaluate_solution(neighbor)
            if neighbor_score > current_score:
                current_solution = neighbor
                current_score = neighbor_score
                no_improve = 0
            else:
                no_improve += 1
            # if no_improve >= max_no_improve:
            #     break
            now = time.perf_counter()
            elapsed_time = now - start_time
            saved_scores.append([current_score, elapsed_time])
        if current_score > best_score:
            best_score = current_score
            best_solution = current_solution
    return best_solution, best_score, saved_scores


def simulated_annealing(runs=3000, max_no_improve=3, random_starts=1, initial_temperature=100, cooling_rate=0.99):
    """
    Simulated Annealing algorithm
    """
    best_solution = []
    best_score = 0
    saved_scores = []

    start_time = time.perf_counter()
    for _ in range(random_starts):
        current_solution = generate_candidate_solution()
        current_score = evaluate_solution(current_solution)

        no_improve = 0
        temperature = initial_temperature

        for x in range(runs):
            neighbor = generate_neighbor_solution(current_solution)
            neighbor_score = evaluate_solution(neighbor)

            if neighbor_score > current_score:
                current_solution = neighbor
                current_score = neighbor_score
                no_improve = 0
            else:
                delta = neighbor_score - current_score
                probability = math.exp(delta / temperature)
                if random.random() < probability:
                    current_solution = neighbor
                    current_score = neighbor_score
                    no_improve = 0
                else:
                    no_improve += 1

            temperature *= cooling_rate
            # if no_improve >= max_no_improve or temperature <= 0.1:
            #     break
            now = time.perf_counter()
            elapsed_time = now - start_time
            saved_scores.append([current_score, elapsed_time])

        if current_score > best_score:
            best_score = current_score
            best_solution = current_solution

    return best_solution, best_score, saved_scores


def generate_population(population_size):
    """
    Generate an initial population of candidate solutions.
    """
    return [generate_candidate_solution() for _ in range(population_size)]


def select_parents(population, tournament_size):
    """
    Select parents for crossover using tournament selection.
    """
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=evaluate_solution)


def crossover(parent1, parent2):
    """
    Perform crossover between two parents to produce two offspring.
    """
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(solution, mutation_rate):
    """
    Mutate a solution by flipping bits with a certain probability.
    """
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = 1 - solution[i]
    return solution


def genetic_algorithm(population_size, generations, mutation_rate, tournament_size):
    """
    Genetic algorithm for the given problem.
    """
    saved_scores = []
    population = generate_population(population_size)
    
    start_time = time.perf_counter()  # Start timing

    for x in range(generations):
        new_population = []
        
        while len(new_population) < population_size:
            parent1 = select_parents(population, tournament_size)
            parent2 = select_parents(population, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population
        now = time.perf_counter()
        elapsed_time = now - start_time
        saved_scores.append([evaluate_solution(max(population, key=evaluate_solution)), elapsed_time])
    
    best_solution = max(population, key=evaluate_solution)
    best_score = evaluate_solution(best_solution)
    return best_solution, best_score, saved_scores


def estimate_distribution(samples):
    """
    Estimate the distribution of features from the provided samples.
    """
    num_features = len(samples[0])
    distributions = []
    for i in range(num_features):
        feature_values = [sample[i] for sample in samples]
        # feature_distribution = sum(feature_values)
        # distributions.append(feature_distribution)
        hist, bin_edges = np.histogram(feature_values, bins=10, density=True)
        # distributions.append((hist, bin_edges))
        distributions.append(hist)
        # distributions.append(feature_distribution)
    return distributions


def generate_sample(distribution):
    """
    Generate a sample based on the given distribution.
    """
    return [int(random.random() < p) for p in distribution]


def mimic(population_size, sample_size, generations):
    """
    MIMIC algorithm for the given problem.
    """
    population = generate_population(population_size)
    saved_scores = []
    start_time = time.perf_counter()
    
    for _ in range(generations):
        samples = random.sample(population, sample_size)
        distribution = estimate_distribution(samples)
        population = [generate_sample(distribution) for _ in range(population_size)]
        now = time.perf_counter()
        elapsed_time = now - start_time
        saved_scores.append([evaluate_solution(max(population, key=evaluate_solution)), elapsed_time])
    
    best_solution = max(population, key=evaluate_solution)
    best_score = evaluate_solution(best_solution)
    
    return best_solution, best_score, saved_scores


def generate_neighbor_solution(solution):
    """
    Generate a neighbor solution by flipping a random bit
    """
    neighbor = solution.copy()
    idx = np.random.randint(len(solution))
    neighbor[idx] = 1 if neighbor[idx] == 0 else 0
    return neighbor


def select_top_individuals(population, scores, keep_percent):
    """
    Select the top individuals to keep based on the given scores.
    """
    sorted_indices = np.argsort(scores)[::-1]
    keep_size = int(len(population) * keep_percent)
    return [population[i] for i in sorted_indices[:keep_size]], [scores[i] for i in sorted_indices[:keep_size]]


def mimic(pop_size=100, keep_percent=0.2, generations=1000):
    """
    MIMIC algorithm for the given problem.
    """
    # Initialize population
    population = generate_population(pop_size)

    # Track best solution and score
    best_solution = None
    best_score = 0

    # Track score over iterations
    saved_scores = []

    start_time = time.time()
    for i in range(generations):
        scores = np.array([evaluate_solution(sol) for sol in population])

        # Update best solution and score
        max_score_idx = np.argmax(scores)
        if scores[max_score_idx] > best_score:
            best_score = scores[max_score_idx]
            best_solution = population[max_score_idx]

        # Select top individuals to keep
        population, _ = select_top_individuals(population, scores, keep_percent)

        # Generate new samples
        new_population = []
        for _ in range(pop_size):
            parent_idx = np.random.randint(len(population))
            parent = population[parent_idx]
            neighbor = generate_neighbor_solution(parent)
            new_population.append(neighbor)
        population = np.array(new_population)

        # Record score at each iteration
        saved_scores.append([best_score, time.time() - start_time])

    return best_solution, best_score, saved_scores

RUN = False

if RUN:
    # Run the hill climbing algorithm 100 times
    rhc_scores = []
    simulated_annealing_scores = []
    genetic_algorithm_scores = []
    mimic_scores = []

    rhc_times = []
    simulated_annealing_times = []
    genetic_algorithm_times = []
    mimic_times = []
    

    for _ in range(100):
        random.seed(_)
        print("Running iteration", _)  # Print iteration number
        _, _, saved_scores = hill_climbing(runs=1500)
        rhc_scores.append([x[0] for x in saved_scores])
        rhc_times.append([x[1] for x in saved_scores])
        _, _, saved_scores = simulated_annealing(runs=1500)
        simulated_annealing_scores.append([x[0] for x in saved_scores])
        simulated_annealing_times.append([x[1] for x in saved_scores])
        _, _, saved_scores = genetic_algorithm(population_size=100, generations=500, mutation_rate=0.02, tournament_size=4)
        genetic_algorithm_scores.append([x[0] for x in saved_scores])
        genetic_algorithm_times.append([x[1] for x in saved_scores])
        _, _, saved_scores = mimic(pop_size=100, keep_percent=0.3, generations=500)
        mimic_scores.append([x[0] for x in saved_scores])
        mimic_times.append([x[1] for x in saved_scores])


    # Calculate mean and standard deviation
    rhc_mean_scores = np.mean(rhc_scores, axis=0)
    rhc_std_dev = np.std(rhc_scores, axis=0)
    rhc_times = np.mean(rhc_times, axis=0)
    simulated_annealing_mean_scores = np.mean(simulated_annealing_scores, axis=0)
    simulated_annealing_std_dev = np.std(simulated_annealing_scores, axis=0)
    simulated_annealing_times = np.mean(simulated_annealing_times, axis=0)
    genetic_algorithm_mean_scores = np.mean(genetic_algorithm_scores, axis=0)
    genetic_algorithm_std_dev = np.std(genetic_algorithm_scores, axis=0)
    genetic_algorithm_times = np.mean(genetic_algorithm_times, axis=0)
    mimic_mean_scores = np.mean(mimic_scores, axis=0)
    mimic_std_dev = np.std(mimic_scores, axis=0)
    mimic_times = np.mean(mimic_times, axis=0)



    # Plotting vs runs
    pdb.set_trace()
    plt.figure(figsize=(10, 6))
    plt.plot(rhc_mean_scores, label='RHC', color='blue')
    plt.fill_between(range(len(rhc_mean_scores)), rhc_mean_scores - rhc_std_dev, rhc_mean_scores + rhc_std_dev, color='lightblue', alpha=0.5)
    plt.plot(simulated_annealing_mean_scores, label='SA', color='orange')
    plt.fill_between(range(len(simulated_annealing_mean_scores)), simulated_annealing_mean_scores - simulated_annealing_std_dev, simulated_annealing_mean_scores + simulated_annealing_std_dev, color='lightcoral', alpha=0.5)
    plt.plot(genetic_algorithm_mean_scores, label='GA', color='green')
    plt.fill_between(range(len(genetic_algorithm_mean_scores)), genetic_algorithm_mean_scores - genetic_algorithm_std_dev, genetic_algorithm_mean_scores + genetic_algorithm_std_dev, color='lightgreen', alpha=0.5)
    plt.plot(mimic_mean_scores, label='MIMIC', color='red')
    plt.fill_between(range(len(mimic_mean_scores)), mimic_mean_scores - mimic_std_dev, mimic_mean_scores + mimic_std_dev, color='pink', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Algorithms based on Iteration')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("pngs/4_peaks_runs_long.png")
    # plt.savefig("pngs/4_peaks_runs.png")
    plt.close()


    # Plotting vs time
    plt.figure(figsize=(10, 6))
    plt.plot(rhc_times, rhc_mean_scores, label='RHC', color='blue')
    plt.fill_between(rhc_times, rhc_mean_scores - rhc_std_dev, rhc_mean_scores + rhc_std_dev, color='lightblue', alpha=0.5)
    plt.plot(simulated_annealing_times, simulated_annealing_mean_scores, label='SA', color='orange')
    plt.fill_between(simulated_annealing_times, simulated_annealing_mean_scores - simulated_annealing_std_dev, simulated_annealing_mean_scores + simulated_annealing_std_dev, color='lightcoral', alpha=0.5)
    plt.plot(genetic_algorithm_times, genetic_algorithm_mean_scores, label='GA', color='green')
    plt.fill_between(genetic_algorithm_times, genetic_algorithm_mean_scores - genetic_algorithm_std_dev, genetic_algorithm_mean_scores + genetic_algorithm_std_dev, color='lightgreen', alpha=0.5)
    plt.plot(mimic_times, mimic_mean_scores, label='MIMIC', color='red')
    plt.fill_between(mimic_times, mimic_mean_scores - mimic_std_dev, mimic_mean_scores + mimic_std_dev, color='pink', alpha=0.5)
    plt.xlabel('Time (seconds)')
    plt.xscale('log')
    plt.ylabel('Score')
    plt.title('Algorithm based on Time (seconds)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("pngs/4_peaks_times_long.png")
    # plt.savefig("pngs/4_peaks_times.png")
    plt.close()



total_rhc_scores = []
total_simulated_annealing_scores = []
total_genetic_algorithm_scores = []
total_mimic_scores = []
total_rhc_std_dev = []
total_simulated_annealing_std_dev = []
total_genetic_algorithm_std_dev = []
total_mimic_std_dev = []


LENGTH_RUN = True

if LENGTH_RUN:
    max_lengths = 200

    start = time.time()

    for prob_l in range(10, max_lengths):
        LENGTH = prob_l
        THRESHOLD = int(0.2 * LENGTH)
        now = time.time()
        elapsed = now - start
        print(f"Running for length {LENGTH}")
        print(f"Elapsed time: {elapsed}")
        rhc_scores = []
        simulated_annealing_scores = []
        genetic_algorithm_scores = []
        mimic_scores = []
        
        for _ in range(10):
            _, _, saved_scores = hill_climbing(runs=500)
            rhc_scores.append([x[0] for x in saved_scores])
            # rhc_times.append([x[1] for x in saved_scores])
            _, _, saved_scores = simulated_annealing(runs=500)
            simulated_annealing_scores.append([x[0] for x in saved_scores])
            # simulated_annealing_times.append([x[1] for x in saved_scores])
            _, _, saved_scores = genetic_algorithm(population_size=100, generations=200, mutation_rate=0.02, tournament_size=4)
            genetic_algorithm_scores.append([x[0] for x in saved_scores])
            # genetic_algorithm_times.append([x[1] for x in saved_scores])
            _, _, saved_scores = mimic(pop_size=100, keep_percent=0.3, generations=200)
            mimic_scores.append([x[0] for x in saved_scores])
            # mimic_times.append([x[1] for x in saved_scores])
        total_rhc_scores.append(np.mean([x[-1] for x in rhc_scores]))
        total_simulated_annealing_scores.append(np.mean([x[-1] for x in simulated_annealing_scores]))
        total_genetic_algorithm_scores.append(np.mean([x[-1] for x in genetic_algorithm_scores]))
        total_mimic_scores.append(np.mean([x[-1] for x in mimic_scores]))
        total_rhc_std_dev.append(np.std([x[-1] for x in rhc_scores]))
        total_simulated_annealing_std_dev.append(np.std([x[-1] for x in simulated_annealing_scores]))
        total_genetic_algorithm_std_dev.append(np.std([x[-1] for x in genetic_algorithm_scores]))
        total_mimic_std_dev.append(np.std([x[-1] for x in mimic_scores]))


    total_rhc_scores = np.array(total_rhc_scores)
    total_simulated_annealing_scores = np.array(total_simulated_annealing_scores)
    total_genetic_algorithm_scores = np.array(total_genetic_algorithm_scores)
    total_mimic_scores = np.array(total_mimic_scores)
    total_rhc_std_dev = np.array(total_rhc_std_dev)
    total_simulated_annealing_std_dev = np.array(total_simulated_annealing_std_dev)
    total_genetic_algorithm_std_dev = np.array(total_genetic_algorithm_std_dev)
    total_mimic_std_dev = np.array(total_mimic_std_dev)
        
    # Plotting vs time
    lengths = list(range(10, max_lengths))
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, total_rhc_scores, label='RHC', color='blue')
    plt.fill_between(lengths, total_rhc_scores -  total_rhc_std_dev, total_rhc_scores + total_rhc_std_dev, color='lightblue', alpha=0.5)
    plt.plot(lengths, total_simulated_annealing_scores, label='SA', color='orange')
    plt.fill_between(lengths, total_simulated_annealing_scores - total_simulated_annealing_std_dev, total_simulated_annealing_scores + total_simulated_annealing_std_dev, color='lightcoral', alpha=0.5)
    plt.plot(lengths, total_genetic_algorithm_scores, label='GA', color='green')
    plt.fill_between(lengths, total_genetic_algorithm_scores - total_genetic_algorithm_std_dev, total_genetic_algorithm_scores + total_genetic_algorithm_std_dev, color='lightgreen', alpha=0.5)
    plt.plot(lengths, total_mimic_scores, label='MIMIC', color='red')
    plt.fill_between(lengths, total_mimic_scores - total_mimic_std_dev, total_mimic_scores + total_mimic_std_dev, color='pink', alpha=0.5) 
    plt.xlabel('Length of problem')
    plt.ylabel('Score')
    plt.title('Algorithm vs. Length of problem')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("pngs/4_peaks_length.png")
    plt.close()