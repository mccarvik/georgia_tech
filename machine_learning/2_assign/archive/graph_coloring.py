import numpy as np
from mlrose_hiive import MIMICRunner, SARunner, GARunner, RHCRunner, DiscreteOpt

# Define the Graph Coloring fitness function
def graph_coloring_fitness(state, adj_matrix):
    conflicts = 0
    num_vertices = len(state)
    for i in range(num_vertices):
        for j in range(num_vertices):
            if adj_matrix[i][j] and state[i] == state[j]:
                conflicts += 1
    return -conflicts  # Minimize conflicts (maximize negative conflicts)

# Function to generate random adjacency matrix for graph
def generate_random_adjacency_matrix(num_vertices, edge_prob):
    adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if np.random.rand() < edge_prob:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
    return adj_matrix

# Define parameters
num_vertices = 10
edge_prob = 0.5
num_trials = 10
pop_size = 200
keep_pct = 0.2
max_iters = 1000

# Generate random adjacency matrix for graph
adj_matrix = generate_random_adjacency_matrix(num_vertices, edge_prob)

# Define optimization problem
problem = DiscreteOpt(length=num_vertices, fitness_fn=lambda x: graph_coloring_fitness(x, adj_matrix), maximize=True, max_val=num_vertices)

# Run optimization algorithms multiple times
algorithms = ['MIMIC', 'Hill Climbing', 'Simulated Annealing', 'Genetic Algorithm']
runners = [MIMICRunner, RHCRunner, SARunner, GARunner]
best_fitness_dict = {algo: [] for algo in algorithms}

for algo, runner in zip(algorithms, runners):
    for _ in range(num_trials):
        if algo == 'MIMIC':
            runner_instance = runner(problem=problem,
                                     experiment_name='graph_coloring_' + algo.lower(),
                                     seed=None,
                                     iteration_list=[max_iters],
                                     population_sizes=[pop_size],
                                     keep_percent_list=[keep_pct],
                                     use_fast_mimic=True)
        else:
            runner_instance = runner(problem=problem,
                                     experiment_name='graph_coloring_' + algo.lower(),
                                     seed=None,
                                     iteration_list=[max_iters])
            
        stats, _ = runner_instance.run()
        best_fitness_dict[algo].append(stats['Fitness'].max())

# Print best fitness obtained from each trial for each algorithm
for algo in algorithms:
    print(f"Best fitness from each trial for {algo}: {best_fitness_dict[algo]}")