# COMP5400M Assignment 2
# Script that Geneerates sudokus and solves them with a Hopfield Network
#
# This implementation is base on the following paper:
# Bauckhage C, Beaumont F, Müller S. ML2R Coding Nuggets Hopfield Nets for Sudoku.
#
# Author: Emmanuel Leo

# Import Libraries
import numpy as np
import numpy.random as rnd
import random

# Puzzle generator and solver functions

# Defining Test Variables
num_of_zeros = 17  # Specifies the number of zeros the problem should have
num_puzzles = 50  # Specifes the number of puzzles to test the solver with test on

# Definition of the Quadratic Unconstrained Binary Optimisation (QUBO) Problem that describes the sudoku problem
# (Based on the paper acknowledged above)


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

    # Create some vectors and matrixrices for later use
    vec1_m = np.ones(m)
    vec1_n1 = np.ones(n1)
    vec1_n2 = np.ones(n2)
    vec1_n3 = np.ones(n3)
    vec1_nh = np.ones(nh)
    matrixI_m = np.eye(m)
    matrixI_n1 = np.eye(n1)
    matrixI_n2 = np.eye(n2)

    # Create the matrix for hint constraints
    h = 0
    matrixH = np.zeros((nh, n3))
    for i in range(n1):
        for j in range(n1):
            v = sudoku[i, j]
            if v > 0:
                # For each hint, set the corresponding row in the matrix to a one-hot encoding of the hint value
                matrixH[h, i * n2 + j * n1: i * n2 +
                        j * n1 + n1] = matrixI_n1[v - 1]
                h = h + 1

    # Create the matrixrices for cell, row, column, and block constraints
    matrixG = np.kron(matrixI_n2, vec1_n1)
    matrixR = np.kron(vec1_n1, matrixI_n1)
    matrixR = np.kron(matrixI_n1, matrixR)
    matrixC = np.kron(vec1_n1, matrixI_n2)
    matrixB = np.kron(vec1_m, matrixI_n1)
    matrixB = np.kron(matrixI_m, matrixB)
    matrixB = np.kron(vec1_m, matrixB)
    matrixB = np.kron(matrixI_m, matrixB)

    # Create the matrix and vector for binary QUBO
    matrixP = lH * matrixH.T @ matrixH
    matrixP += lG * matrixG.T @ matrixG
    matrixP += lR * matrixR.T @ matrixR
    matrixP += lC * matrixC.T @ matrixC
    matrixP += lB * matrixB.T @ matrixB
    vecP = lH * 2 * vec1_nh @ matrixH
    vecP += lG * 2 * vec1_n2 @ matrixG
    vecP += lR * 2 * vec1_n2 @ matrixR
    vecP += lC * 2 * vec1_n2 @ matrixC
    vecP += lB * 2 * vec1_n2 @ matrixB

    # Create the matrix and vector for bipolar QUBO
    matrixQ = 0.25 * matrixP
    vecQ = 0.50 * (matrixP @ vec1_n3 - vecP)

    # Create the weight matrix
    matrixW = -2 * matrixQ
    np . fill_diagonal(matrixW, 0)  # Set diagonal elements to zero
    vecT = vecQ
    return matrixW, vecT

# Function that performs simulated annealing (Based on paper acknowledged above)


def hnetSimAnn(state_vector, matrixW, vecT, max_Temp=10, min_Temp=0.5, numT=21, rmax=200):  # Define
    n = len(state_vector)
    for T in np.linspace(max_Temp, min_Temp, numT):
        for r in range(rmax):
            for i in range(n):
                q = 1 / (1 + np . exp(-2 / T *
                         (matrixW[i] @ state_vector - vecT[i])))
                z = rnd . binomial(n=1, p=q)
                state_vector[i] = 2 * z - 1
    return state_vector

# Generates a 4 by 4 sudoku puzzle from the first 4 non-zero digits


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

# Randomly sets the specified number of cells in grid to zero


def makePuzzle(grid, num_zeros):
    puzzle = np.copy(grid)
    n = len(puzzle)
    for _ in range(num_zeros):
        i = np.random.randint(n)
        j = np.random.randint(n)
        puzzle[i][j] = 0
    return np.array(puzzle)

# Wrapper function that generates puzzles from a solution


def testingPuzzleGenerator(num_of_zeros):
    # print(f"Puzzle {i+1}:")
    solution = genPuzzle()
    puzzle = makePuzzle(solution, num_of_zeros)
    # Print Puzzle
    # for row in puzzle:
    #     print(row)
    # print()
    # print(puzzle)

    return puzzle, solution

# Prints the sudoku as an n by n grid of un-encoded digits


def print_board(sudoku):
    n = int(np.cbrt(len(sudoku)))
    for i in range(n*n):
        for j in range(n):
            if(sudoku[i*n+j] == 1):
                print(j+1, end='   ')
            if (j+1) % n == 0:
                print('|', end=' ')
        if (i+1) % n == 0:
            print()
            print('- ' * (n*n - n))

# Takes the sudoku solution in the one-hot encoding formatrix and converts it back to the digits of the puzzle


def decode(sudoku):
    m = int(np.cbrt(len(sudoku)))
    solution = np.ones(m*m, dtype=int)
    for i in range(m*m):
        for j in range(m):
            if(sudoku[i*m+j] == 1):
                solution[i] = j+1
    # print(solution.reshape(4, 4))
    return solution.reshape(4, 4)

# Calculates the accuracy and completeness scores of the function


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
for zeros in range(num_of_zeros):
    print(f"number of zeros in puzzle is: ", end=' ')
    print(zeros)
    puzzles = []
    solutions = []
    for i in range(num_puzzles):
        puzzle, solution = testingPuzzleGenerator(zeros)
    # Uncomment to view generated solutions and puzles for testin
        # print(puzzle.reshape(4,4))
        # print(solution.reshape(4,4))
        puzzles.append(puzzle)
        solutions.append(solution)

    # solve each puzzle and calculate completeness and accuracy
    total_accuracy = 0
    total_completeness = 0
    for i in range(num_puzzles):
        matrixW_i, vecT_i = hnetInitParameters(puzzles[i])
        n3 = puzzles[i].shape[0]**3
        state_vector_i = rnd.binomial(n=1, p=0.05, size=n3)*2-1
        solution_attempt = hnetSimAnn(state_vector_i, matrixW_i, vecT_i)
        # print_board(solution_attempt)
        solution_attempt = decode(solution_attempt)
        accuracy, completeness = evaluate_sudoku_solver(
            solution_attempt, np.array(solutions[i]))
        total_accuracy += accuracy
        total_completeness += completeness

    total_accuracy = total_accuracy/num_puzzles
    total_completeness = total_completeness/num_puzzles
    print(total_accuracy)
    print(total_completeness)
