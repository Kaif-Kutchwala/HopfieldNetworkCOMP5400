import numpy as np
from chn_solver import CHNSudokuSolver

class PSOSudokuSolver(CHNSudokuSolver):
    def __init__(self, weights=None, max_iter=10000, pop_size=100, c1=2.0, c2=2.0, w=0.7):
        super().__init__(weights)
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
    
    def solve_sudoku(self, grid):
        # Initialize the particle swarm
        swarm = np.random.rand(self.pop_size, self.num_neurons) * 2.0 - 1.0
        best_swarm = np.zeros_like(swarm)
        best_fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            best_swarm[i] = swarm[i]
            best_fitness[i] = self.calculate_fitness(swarm[i])
        
        # Loop until the maximum number of iterations is reached
        for i in range(self.max_iter):
            # Update the velocity and position of each particle
            for j in range(self.pop_size):
                velocity = self.w * swarm[j] + self.c1 * np.random.rand() * (best_swarm[j] - swarm[j]) + self.c2 * np.random.rand() * (self.best_global - swarm[j])
                swarm[j] = np.clip(swarm[j] + velocity, -1.0, 1.0)
            
            # Update the best solutions found by each particle and the global best solution
            for j in range(self.pop_size):
                fitness = self.calculate_fitness(swarm[j])
                if fitness > best_fitness[j]:
                    best_swarm[j] = swarm[j]
                    best_fitness[j] = fitness
                if fitness > self.best_fitness:
                    self.best_global = swarm[j]
                    self.best_fitness = fitness
            
            # Check if a valid solution has been found
            if self.is_valid_solution(self.best_global):
                return self.vector_to_sudoku(self.best_global)
        
        # If no valid solution is found, return None
        return None
    
    def calculate_fitness(self, solution):
        # Calculate the energy of the solution vector
        energy = np.dot(solution, np.dot(self.weights, solution)) / -2.0
        
        # Convert the solution vector to a Sudoku grid
        grid = self.vector_to_sudoku(solution)
        
        # Calculate the number of conflicts in the Sudoku grid
        conflicts = self.count_conflicts(grid)
        
        # Calculate the fitness score as the inverse of the energy plus the number of conflicts
        return 1.0 / (energy + conflicts + 1)
