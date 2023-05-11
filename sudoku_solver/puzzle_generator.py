import random
import numpy as np


def genPuzzle():
    nums = [1, 2, 3, 4]
    n = 4
    grid = [[0 for x in range(n)] for y in range(n)]
    for i in range(n):
        for j in range(n):
            choices = [x for x in nums if x not in grid[i]
                       and x not in [grid[k][j] for k in range(n)]]
            if len(choices) == 0:
                return genPuzzle()
            grid[i][j] = random.choice(choices)
    return grid


def makePuzzle(grid, num_zeros):
    n = len(grid)
    for _ in range(num_zeros):
        i = np.random.randint(n)
        j = np.random.randint(n)
        grid[i][j] = 0
    return np.array(grid)


for i in range(10):
    print(f"Puzzle {i+1}:")
    puzzle = genPuzzle()
    puzzle = makePuzzle(puzzle, 8)

    for row in puzzle:
        print(row)
    print()
