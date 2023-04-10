import numpy as np
from chn_solver import CHNSudokuSolver

# Define a Sudoku puzzle as a numpy array
grid = np.array([
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

# Create an instance of the CHNSudokuSolver class
solver = CHNSudokuSolver()

# Solve the Sudoku puzzle using the CHN solver
solution = solver.solve_sudoku(grid)

# Print the solution
print(solution)
