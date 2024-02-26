"""
4 peaks problem
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, random, math, pdb


# # Define the objective function (Four Peaks Problem)
# def four_peaks_obj(x, t1, t2):
#     if 0 <= x < t1:
#         return x + 0.5 * math.sin(5 * math.pi * x)
#     elif t1 <= x < t2:
#         return 1 + math.sin(10 * math.pi * (x - t1)) * 0.2
#     elif t2 <= x <= 1:
#         return 1 - (x - t2) ** 2

# plot_four_peaks(0.25,0.75)

# Example usage
peaks = [0.03, 0.8, 0.2, 0.7, 0.5, 0.35, 0.90, 0.75]  # [peak1_x, peak1_y, peak2_x, peak2_y, ..., peak4_x, peak4_y]


def four_peaks_obj(x, peaks):
    """
    Objective function for the Four Peaks Problem with user-defined peaks.
    """
    num_peaks = len(peaks) // 2
    y_values = []
    for i in range(num_peaks):
        peak_x = peaks[2*i]
        peak_y = peaks[2*i + 1]
        y_values.append(peak_y * np.exp(-((x - peak_x) ** 2) / 0.05))  # Gaussian-like peaks
    return max(y_values)


def plot_four_peaks():
    """
    Plot the Four Peaks objective function.
    """
    x_values = np.linspace(0, 1, 1000)
    y_values = [four_peaks_obj(xi, peaks) for xi in x_values]
    plt.plot(x_values, y_values)
    plt.title('Four Peaks Objective Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.savefig("pngs/4peaks.png")
    plt.close()


def evaluate_solution(solution):
    """
    Evaluate a solution by finding the highest peak it reaches.
    """
    highest_peak = max(peaks[::2])
    # index = solution - 1  # adjusting for 0-based indexing
    return max(0, highest_peak - solution)


def generate_random_solution():
    """
    Generate a random solution.
    """
    return round(random.uniform(0, 1), 2)


def hill_climbing(rand_starts=1, step_size = 0.01, verbose=False):
    """
    Perform hill climbing algorithm to find the highest peak.
    """
    best_solution = None
    best_value = 0

    for rs in range(rand_starts):
        print("Random start: ", rs+1)
        current_solution = generate_random_solution()
        current_value = four_peaks_obj(current_solution, peaks)
        print("original location:", current_solution, "    original value: ", current_value)

        steps = 1
        while True:
            if verbose:
                print("step: ", steps)
            right_higher = four_peaks_obj(current_solution + step_size, peaks) > current_value
            left_higher = four_peaks_obj(current_solution - step_size, peaks) > current_value
            if not right_higher and not left_higher:
                break
            if right_higher and left_higher:
                if random.random() > 0.5:
                    # choose right
                    current_solution = current_solution + step_size
                    current_value = four_peaks_obj(current_solution, peaks)
                else:
                    # choose left
                    current_solution = current_solution - step_size
                    current_value = four_peaks_obj(current_solution, peaks)
            elif right_higher:
                current_solution = current_solution + step_size
                current_value = four_peaks_obj(current_solution, peaks)
            elif left_higher:
                current_solution = current_solution - step_size
                current_value = four_peaks_obj(current_solution, peaks)
            
            steps += 1
            if verbose:
                print("current location: ", current_solution, "   current value: ", current_value)
        
        print("iteration {} solution:" .format(rs+1), current_solution)
        print("iteration {} value:" .format(rs+1), current_value)
        print("steps: ", steps)
        if current_value > best_value:
            best_solution = current_solution
            best_value = current_value

        print("final location: ", best_solution, "    final value: ", best_value)
        print("distance from peak: ", evaluate_solution(best_value))
    return best_solution, best_value


def simulated_annealing(rand_starts=1, initial_temperature=100.0, step_size=0.01, cooling_rate=0.99, verbose=False):
    """
    Perform simulated annealing algorithm to find the highest peak.
    """
    best_solution = None
    best_value = 0

    for rs in range(rand_starts):
        print("Random start: ", rs+1)
        current_solution = generate_random_solution()  # Random initial solution
        current_value = four_peaks_obj(current_solution, peaks)
        temperature = initial_temperature
        print("Original location:", current_solution, "    Original value:", current_value)

        steps = 1
        while temperature > 1:
            if verbose:
                print("Step:", steps)
            
            # check if we accept a random better solution or random exploration option
            pot_neighbor = round(random.uniform(0, 1), 2)  # Generate a random neighbor
            pot_neighbor_value = four_peaks_obj(pot_neighbor, peaks)
            delta = pot_neighbor_value - current_value
            if delta > 0 or np.random.rand() < math.exp(delta / temperature):
                current_solution = pot_neighbor
                current_value = pot_neighbor_value
            else:
                right_higher = four_peaks_obj(current_solution + step_size, peaks) > current_value
                left_higher = four_peaks_obj(current_solution - step_size, peaks) > current_value
                # If at local or global optimum, do nothing, lower temperature
                if not right_higher and not left_higher:
                    pass
                if right_higher and left_higher:
                    if random.random() > 0.5:
                        # choose right
                        current_solution = current_solution + step_size
                        current_value = four_peaks_obj(current_solution, peaks)
                    else:
                        # choose left
                        current_solution = current_solution - step_size
                        current_value = four_peaks_obj(current_solution, peaks)
                elif right_higher:
                    current_solution = current_solution + step_size
                    current_value = four_peaks_obj(current_solution, peaks)
                elif left_higher:
                    current_solution = current_solution - step_size
                    current_value = four_peaks_obj(current_solution, peaks)
            
            if verbose:
                print("Current location:", current_solution, "   Current value:", current_value)
                print("Temperature:", temperature)
            
            temperature *= cooling_rate
            steps += 1
        
        print("Iteration {} solution:".format(rs+1), current_solution)
        print("Iteration {} value:".format(rs+1), current_value)
        print("Steps:", steps)
        
        if current_value > best_value:
            best_solution = current_solution
            best_value = current_value

        print("Final location:", best_solution, "    Final value:", best_value)
        print("Distance from peak:", evaluate_solution(best_value))

    return best_solution, best_value


def genetic_algorithm(population_size=100, generations=100, mutation_rate=0.01, verbose=False):
    """
    Perform genetic algorithm to find the highest peak.
    """
    # Create the initial population
    population = []
    for i in range(population_size):
        population.append(generate_random_solution())

    # How many winners from each generation?
    top_limit = int(population_size * 0.2)

    for generation in range(generations):
        if verbose:
            print("Generation:", generation)
        scores = [(four_peaks_obj(population[i], peaks), population[i]) for i in range(population_size)]
        scores.sort(reverse=True)
        if verbose:
            print("Scores:", scores)
        top_solutions = [x[1] for x in scores[:top_limit]]

        # Create children
        children = []
        while len(children) < population_size:
            if verbose:
                print("Children:", children)


if __name__ == "__main__":
    plot_four_peaks()

    # Example usage
    best_solution, best_value = hill_climbing()
    print("Best solution:", best_solution)
    print("Best value:", best_value)

    best_solution, best_value = simulated_annealing()
    print("Best solution:", best_solution)
    print("Best value:", best_value)
