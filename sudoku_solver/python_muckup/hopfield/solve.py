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

# Create an instance of the CHNSudokuSolver class (Un-Comment to Use the Hopfield Network on it's own)
solver = CHNSudokuSolver()

# Create a solver object (Un-Comment to Use the Hopfield Network with Genetic Algorithm to Optimise the Solution Vector)
# solver = GASudokuSolver()

# Create a solver object (Un-Comment to Use the Hopfield Network with Particle Swarm Optimisation to Optimise the Solution Vector)
# solver = PSOSudokuSolver()

# Define a Sudoku grid as a string
# grid = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'

# Solve the Sudoku puzzle
solution = solver.solve_sudoku(grid)

# Print the solution
print(solution)


# Solve the Sudoku puzzle using the CHN solver
solution = solver.solve_sudoku(grid)

# Print the solution
print(solution)
