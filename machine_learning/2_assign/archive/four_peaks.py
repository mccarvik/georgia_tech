"""
Four peaks problem
"""

import math
import random

LENGTH = 100
THRESHOLD = 7

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
        return max(num_zeros, num_ones) + 100
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


def hill_climbing(max_no_improve=3, random_starts=1):
    """
    Hill climbing algorithm
    """
    best_solution = 0
    best_score = 0

    for _ in range(random_starts):
        current_solution = generate_candidate_solution()
        current_score = evaluate_solution(current_solution)

        no_improve = 0
        while True:
            neighbor = generate_neighbor_solution(current_solution)
            neighbor_score = evaluate_solution(neighbor)
            if neighbor_score > current_score:
                current_solution = neighbor
                current_score = neighbor_score
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= max_no_improve:
                break
        if current_score > best_score:
            best_score = current_score
            best_solution = current_solution
    return best_solution, best_score


def simulated_annealing(max_no_improve=3, random_starts=1, initial_temperature=100, cooling_rate=0.99):
    """
    Simulated Annealing algorithm
    """
    best_solution = []
    best_score = 0

    for _ in range(random_starts):
        current_solution = generate_candidate_solution()
        current_score = evaluate_solution(current_solution)

        no_improve = 0
        temperature = initial_temperature

        while True:
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
            if no_improve >= max_no_improve or temperature <= 0.1:
                break

        if current_score > best_score:
            best_score = current_score
            best_solution = current_solution

    return best_solution, best_score


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
    population = generate_population(population_size)
    
    for _ in range(generations):
        new_population = []
        
        while len(new_population) < population_size:
            parent1 = select_parents(population, tournament_size)
            parent2 = select_parents(population, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population
    
    best_solution = max(population, key=evaluate_solution)
    best_score = evaluate_solution(best_solution)
    
    return best_solution, best_score


def estimate_distribution(samples):
    """
    Estimate the distribution of features from the provided samples.
    """
    num_features = len(samples[0])
    distributions = []
    for i in range(num_features):
        feature_values = [sample[i] for sample in samples]
        feature_distribution = sum(feature_values) / len(feature_values)
        distributions.append(feature_distribution)
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
    
    for _ in range(generations):
        samples = random.sample(population, sample_size)
        distribution = estimate_distribution(samples)
        population = [generate_sample(distribution) for _ in range(population_size)]
    
    best_solution = max(population, key=evaluate_solution)
    best_score = evaluate_solution(best_solution)
    
    return best_solution, best_score

