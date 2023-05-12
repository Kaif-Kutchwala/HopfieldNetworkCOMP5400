from sudoku import Sudoku
import numpy as np

# Generate 1000 Sudoku puzzles
puzzles = []
for _ in range(1000):
    puzzle = Sudoku()
    puzzle.generate()
    puzzles.append(puzzle.board)

# Convert puzzles to NumPy array and save to file
puzzles = np.array(puzzles)
np.save("sudoku_training_data.npy", puzzles)