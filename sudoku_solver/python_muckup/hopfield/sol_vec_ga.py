import numpy as np
import random

from chn_solver import CHNSudokuSolver
class GASudokuSolver(CHNSudokuSolver):
    def __init__(self, solver=None):
        super().__init__()
        self.num_neurons = 729
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.pop_size = 100
        self.mutation_rate = 0.01
        self.elite_size = 10
        self.generations = 50

    def solve_sudoku(self, grid):
        # Convert the Sudoku grid to a flat list
        initial_solution = self.sudoku_to_vector(grid)
        
        # Create the initial population
        population = self.create_population(initial_solution)

        for i in range(self.generations):
            # Rank the population
            ranked_pop = self.rank_population(population)

            # Select the parents for the next generation
            selected_pop = self.selection(ranked_pop)

            # Create the next generation
            next_gen_pop = self.create_next_generation(selected_pop)

            # Add the elite individuals to the next generation
            next_gen_pop[0:self.elite_size] = ranked_pop[0:self.elite_size]

            # Set the population for the next generation
            population = next_gen_pop

        # Return the solution with the best fitness score
        return self.vector_to_sudoku(ranked_pop[0][0])

    def create_population(self, solution):
        # Create a list of populations with random values
        population = []
        for i in range(self.pop_size):
            new_individual = solution[:]
            random.shuffle(new_individual)
            population.append(new_individual)
        return population
    
    def calculate_fitness(self, solution):
        """
        Calculates the fitness of a given Sudoku solution, which is the sum of the absolute values
        of the differences between each row, column, and 3x3 subgrid and the set {1, 2, ..., 9}.
        """
        fitness = 0

        # Calculate the fitness for each row
        for i in range(9):
            row_sum = sum(solution[i * 9:(i + 1) * 9])
            fitness += abs(row_sum - 45)

        # Calculate the fitness for each column
        for j in range(9):
            col_sum = sum(solution[j::9])
            fitness += abs(col_sum - 45)

        # Calculate the fitness for each 3x3 subgrid
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid_sum = sum(solution[(i + x) * 9 + j:(i + x) * 9 + j + 3] for x in range(3))
                fitness += abs(subgrid_sum - 45)

        return fitness

    def rank_population(self, population):
        # Rank the population by fitness score
        ranked_pop = []
        for individual in population:
            fitness_score = self.fitness(individual)
            ranked_pop.append((individual, fitness_score))
        ranked_pop.sort(key=lambda x: x[1], reverse=True)
        return ranked_pop

    def selection(self, ranked_pop):
        # Select the parents for the next generation
        selected_pop = []
        total_fitness = sum([x[1] for x in ranked_pop])
        for i in range(self.pop_size):
            rand_num = random.uniform(0, total_fitness)
            running_sum = 0
            for j in range(len(ranked_pop)):
                running_sum += ranked_pop[j][1]
                if running_sum >= rand_num:
                    selected_pop.append(ranked_pop[j][0])
                    break
        return selected_pop

    def create_next_generation(self, selected_pop):
        # Create the next generation
        next_gen_pop = []
        for i in range(self.pop_size):
            parent_1 = random.choice(selected_pop)
            parent_2 = random.choice(selected_pop)
            child = self.crossover(parent_1, parent_2)
            next_gen_pop.append(child)
        return next_gen_pop

    def crossover(self, parent_1, parent_2):
        # Perform uniform crossover
        child = []
        for i in range(len(parent_1)):
            if random.random() < 0.5:
                child.append(parent_1[i])
            else:
                child.append(parent_2[i])
        return child

    def mutation(self, individual):
        # Perform random mutation
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(1, 9)
        return individual

    def fitness(self, individual):
        # Calculate the fitness score
        solution = self.vector_to_sudoku(individual)
        return self.calculate_fitness
