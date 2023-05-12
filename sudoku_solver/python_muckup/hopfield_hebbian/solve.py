import numpy as np
from chn_solver import CHNSudokuSolver
from sol_vec_ga import GASudokuSolver
from sol_vec_particle_swarm import PSOSudokuSolver

# Define a Sudoku puzzle as a numpy array
grid0 = np.array([
    [0, 0, 0, 0, 9, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 6, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 0]
])

grid = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])

grid0_solution = np.array([
    [3, 6, 7, 8, 9, 2, 1, 5, 4],
    [1, 5, 8, 3, 4, 6, 7, 9, 2],
    [9, 4, 2, 1, 5, 7, 3, 8, 6],
    [7, 2, 6, 5, 8, 1, 4, 3, 9],
    [8, 1, 5, 9, 3, 4, 2, 7, 6],
    [4, 9, 0, 2, 7, 3, 5, 6, 1],
    [6, 3, 1, 4, 2, 5, 8, 0, 7],
    [2, 8, 3, 7, 6, 9, 0, 1, 5],
    [5, 7, 9, 6, 1, 8, 0, 2, 3]
])

grid_solution = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9]
])

# Create an instance of the CHNSudokuSolver class (Un-Comment to Use the Hopfield Network on it's own)
solver = CHNSudokuSolver()

# Create a solver object (Un-Comment to Use the Hopfield Network with Genetic Algorithm to Optimise the Solution Vector)
# solver = GASudokuSolver()

# Create a solver object (Un-Comment to Use the Hopfield Network with Particle Swarm Optimisation to Optimise the Solution Vector)
# solver = PSOSudokuSolver()

# Define a Sudoku grid as a string
# grid = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'

solver.train(grid0_solution)

# Solve the Sudoku puzzle using the CHN solver
solution = solver.solve_sudoku(grid0)

# Print the solution
print(solution)
