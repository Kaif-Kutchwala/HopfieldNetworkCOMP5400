from chn_solver import CHNSudokuSolver

class GASudokuSolver(CHNSudokuSolver):
    def __init__(self, weights=None, max_iter=10000, pop_size=100, elite_size=20, mutation_rate=0.05):
        super().__init__(weights)
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
    
    def solve_sudoku(self, grid):
        # Initialize the population
        population = [self.sudoku_to_vector(self.generate_random_sudoku(grid)) for i in range(self.pop_size)]
        
        # Loop until the maximum number of iterations is reached
        for i in range(self.max_iter):
            # Calculate the fitness of each solution vector
            fitness_scores = [self.calculate_fitness(solution) for solution in population]
            
            # Identify the elite solutions
            elite_indices = np.argsort(fitness_scores)[::-1][:self.elite_size]
            elite_population = [population[index] for index in elite_indices]
            
            # Create the next generation of solutions using crossover and mutation
            children = []
            while len(children) < self.pop_size - self.elite_size:
                parent1, parent2 = np.random.choice(elite_population, size=2, replace=False)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                children.append(child)
            
            # Combine the elite solutions with the children to create the next population
            population = elite_population + children
            
            # Check if a valid solution has been found
            for solution in population:
                if self.is_valid_solution(solution):
                    return self.vector_to_sudoku(solution)
        
        # If no valid solution is found, return None
        return None
    
    def crossover(self, parent1, parent2):
        # Perform uniform crossover
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child
    
    def mutate(self, solution):
        # Perform bit-flip mutation
        for i in range(len(solution)):
            if np.random.rand() < self.mutation_rate:
                solution[i] = 1 - solution[i]
        return solution
    
    def calculate_fitness(self, solution):
        # Calculate the energy of the solution vector
        energy = np.dot(solution, np.dot(self.weights, solution)) / -2.0
        
        # Convert the solution vector to a Sudoku grid
        grid = self.vector_to_sudoku(solution)
        
        # Calculate the number of conflicts in the Sudoku grid
        conflicts = self.count_conflicts(grid)
        
        # Calculate the fitness score as the inverse of the energy plus the number of conflicts
        return 1.0 / (energy + conflicts + 1)
