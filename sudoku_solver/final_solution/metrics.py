import random as rnd
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import random

# Define a list of puzzles, each with a solution
puzzles = [
    {
        "puzzle": [0, 2, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 3],
        "solution": [3, 2, 4, 1, 1, 3, 1, 4, 2, 4, 1, 2, 2, 3, 4, 3],
        "difficulty": "easy"
    },
    {
        "puzzle": [0, 2, 4, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 3],
        "solution": [3, 2, 4, 1, 1, 3, 1, 4, 2, 4, 1, 2, 2, 3, 4, 3],
        "difficulty": "medium"
    },
    {
        "puzzle": [0, 2, 4, 0, 1, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3],
        "solution": [3, 2, 4, 1, 1, 3, 1, 4, 2, 4, 1, 2, 2, 3, 4, 3],
        "difficulty": "hard"
    },
    # More puzzles can be added here if needed
]

# Puzzle generator and solver functions


def hnetInitParameters(sudoku, lH=4., lG=2., lR=1., lC=1., lB=1.):
    # Get the size of the sudoku grid
    n1 = sudoku.shape[0]
    # Calculate the total number of cells in the sudoku grid
    n2 = n1 ** 2
    # Calculate the total number of possibilities for each cell in the sudoku grid
    n3 = n1 ** 3
    # Calculate the size of each block in the sudoku grid
    m = np.sqrt(n1).astype(int)
    # Count the number of given hints in the sudoku grid
    nh = np.sum(sudoku > 0)

    # Create some vectors and matrices for later use
    vec1_m = np.ones(m)
    vec1_n1 = np.ones(n1)
    vec1_n2 = np.ones(n2)
    vec1_n3 = np.ones(n3)
    vec1_nh = np.ones(nh)
    matI_m = np.eye(m)
    matI_n1 = np.eye(n1)
    matI_n2 = np.eye(n2)

    # Create the matrix for hint constraints
    h = 0
    matH = np.zeros((nh, n3))
    for i in range(n1):
        for j in range(n1):
            v = sudoku[i, j]
            if v > 0:
                # For each hint, set the corresponding row in the matrix to a one-hot encoding of the hint value
                matH[h, i * n2 + j * n1: i * n2 + j * n1 + n1] = matI_n1[v - 1]
                h = h + 1

    # Create the matrices for cell, row, column, and block constraints
    matG = np.kron(matI_n2, vec1_n1)
    matR = np.kron(vec1_n1, matI_n1)
    matR = np.kron(matI_n1, matR)
    matC = np.kron(vec1_n1, matI_n2)
    matB = np.kron(vec1_m, matI_n1)
    matB = np.kron(matI_m, matB)
    matB = np.kron(vec1_m, matB)
    matB = np.kron(matI_m, matB)

    # Create the matrix and vector for binary QUBO
    matP = lH * matH.T @ matH
    matP += lG * matG.T @ matG
    matP += lR * matR.T @ matR
    matP += lC * matC.T @ matC
    matP += lB * matB.T @ matB
    vecP = lH * 2 * vec1_nh @ matH
    vecP += lG * 2 * vec1_n2 @ matG
    vecP += lR * 2 * vec1_n2 @ matR
    vecP += lC * 2 * vec1_n2 @ matC
    vecP += lB * 2 * vec1_n2 @ matB

    # Create the matrix and vector for bipolar QUBO
    matQ = 0.25 * matP
    vecQ = 0.50 * (matP @ vec1_n3 - vecP)

    # Create the weight matrix
    matW = -2 * matQ
    np . fill_diagonal(matW, 0)  # Set diagonal elements to zero
    vecT = vecQ
    return matW, vecT


def hnetSimAnn(vecS, matW, vecT, Th=10, Tl=0.5, numT=21, rmax=200):
    n = len(vecS)
    for T in np . linspace(Th, Tl, numT):
        for r in range(rmax):
            for i in range(n):
                q = 1 / (1 + np . exp(-2 / T * (matW[i] @ vecS - vecT[i])))
                z = rnd . binomial(n=1, p=q)
                vecS[i] = 2 * z - 1
    return vecS


# def print_board(sudoku):
#     cell_id = 0
#     m = int(np.cbrt(len(sudoku)))
#     solution = np.ones(m*m, dtype=int)
#     for i in range(m*m):
#         for j in range(m):
#             if(sudoku[i*m+j] == 1):
#                 solution[i] = j+1
#                 print(j+1, end='   ')
#             if (j+1) % m == 0:
#                 print('|', end=' ')
#         if (i+1) % m == 0:
#             print()
#             print('- ' * (m*m - m))
#             cell_id = cell_id + 1
#     return solution


def genPuzzle():
    nums = [1, 2, 3, 4]
    n = 4
    grid = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            choices = [x for x in nums if x not in list(grid[i])
                       and x not in [grid[k][j] for k in range(n)]]
            if len(choices) == 0:
                return genPuzzle()
            grid[i][j] = random.choice(choices)
    return grid.astype(int).tolist()


def makePuzzle(grid, num_zeros):
    puzzle = np.copy(grid)
    n = len(puzzle)
    for _ in range(num_zeros):
        i = np.random.randint(n)
        j = np.random.randint(n)
        puzzle[i][j] = 0
    return np.array(puzzle)


def testingPuzzleGenerator(num_of_zeros):
    print(f"Puzzle {i+1}:")
    solution = genPuzzle()
    puzzle = makePuzzle(solution, num_of_zeros)

    for row in solution:
        print(row)
    print()
    print(solution)

    return puzzle, solution


def print_board(sudoku):
    m = int(np.cbrt(len(sudoku)))
    for i in range(m*m):
        for j in range(m):
            if(sudoku[i*m+j] == 1):
                print(j+1, end='   ')
            if (j+1) % m == 0:
                print('|', end=' ')
        if (i+1) % m == 0:
            print()
            print('- ' * (m*m - m))


def decode(sudoku):
    m = int(np.cbrt(len(sudoku)))
    solution = np.ones(m*m, dtype=int)
    for i in range(m*m):
        for j in range(m):
            if(sudoku[i*m+j] == 1):
                solution[i] = j+1
    print(solution.reshape(4, 4))
    return solution.reshape(4, 4)


def evaluate_sudoku_solver(puzzle_list, solution_list):
    num_correct = 0
    num_complete = 0
    for i in range(len(puzzle_list)):
        puzzle = puzzle_list[i]
        solution = solution_list[i]
    # check if the solution is correct
        is_correct = all([solution[i] == puzzle[i]
                         for i in range(len(solution))])
        if is_correct:
            num_correct += 1
    # check if the solution is complete
        is_complete = all([val != 0 for val in solution])
        if is_complete:
            num_complete += 1
    accuracy = num_correct / len(puzzle_list)
    completeness = num_complete / len(puzzle_list)
    return accuracy, completeness


# generate a set of sudoku puzzles
num_of_zeros = 8
num_puzzles = 5
puzzles = []
solutions = []
for i in range(num_puzzles):
    puzzle, solution = testingPuzzleGenerator(num_of_zeros)
    puzzles.append(puzzle)
    solutions.append(solution)

# solve each puzzle and calculate completeness and accuracy
total_accuracy = 0
total_completeness = 0
for i in range(num_puzzles):
    matW_i, vecT_i = hnetInitParameters(puzzles[i])
    n3 = puzzles[i].shape[0]**3
    vecS_i = rnd.binomial(n=1, p=0.05, size=n3)*2-1
    sol_i = hnetSimAnn(vecS_i, matW_i, vecT_i)
    # print_board(sol_i)
    sol_i = decode(sol_i)
    accuracy, completeness = evaluate_sudoku_solver(
        sol_i, np.array(solutions[i]))
    total_accuracy += accuracy
    total_completeness += completeness

total_accuracy = total_accuracy/num_puzzles
total_completeness = total_completeness/num_puzzles
print(total_accuracy)
print(total_completeness)
