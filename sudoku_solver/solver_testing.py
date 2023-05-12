import numpy as np
import numpy.random as rnd
import numpy as np
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
    # Add more puzzles here...
]

# Puzzle generator and solver functions


def hnetInitParameters(sudoku, lH=4., lG=2., lR=1., lC=1., lB=1.):
    n1 = sudoku . shape[0]
    n2 = n1 ** 2
    n3 = n1 ** 3
    m = np.sqrt(n1). astype(int)
    nh = np.sum(sudoku > 0)  # number of hints
    vec1_m = np.ones(m)
    vec1_n1 = np.ones(n1)
    vec1_n2 = np.ones(n2)
    vec1_n3 = np.ones(n3)
    vec1_nh = np.ones(nh)
    matI_m = np.eye(m)
    matI_n1 = np.eye(n1)
    matI_n2 = np.eye(n2)
    # matrix forhnt constraints
    h = 0
    matH = np . zeros((nh, n3))
    for i in range(n1):
        for j in range(n1):
            v = sudoku[i, j]
            if v > 0:
                matH[h, i * n2 + j * n1: i * n2 + j * n1 + n1] = matI_n1[v - 1]
                h = h + 1
    # matrices for cell , row , column , and block constraints
    matG = np.kron(matI_n2, vec1_n1)
    matR = np.kron(vec1_n1, matI_n1)
    matR = np.kron(matI_n1, matR)
    matC = np.kron(vec1_n1, matI_n2)
    matB = np.kron(vec1_m, matI_n1)
    matB = np.kron(matI_m, matB)
    matB = np.kron(vec1_m, matB)
    matB = np.kron(matI_m, matB)
    # matrix and vector for binary QUBO
    matP = lH*matH.T@matH
    matP += lG*matG.T@matG
    matP += lR*matR.T@matR
    matP += lC*matC.T@matC
    matP += lB*matB.T@matB
    vecP = lH*2*vec1_nh@matH
    vecP += lG*2*vec1_n2@matG
    vecP += lR*2*vec1_n2@matR
    vecP += lC*2*vec1_n2@matC
    vecP += lB*2*vec1_n2@matB
    # matrix and vector for bipolar QUBO
    matQ = 0.25 * matP
    vecQ = 0.50 * (matP @ vec1_n3 - vecP)
    # weight matrix and bias vector for Hopfield net
    matW = -2 * matQ
    np . fill_diagonal(matW, 0)
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

# Generate more puzzles by randomly shuffling the numbers in each puzzle


# def genPuzzle():
#     for i in range(50):  # Change 1000 to the number of puzzles you want to generate
#         # Choose a random difficulty level for the new puzzle
#         difficulty = random.choice(["easy", "medium", "hard"])
#         # Choose a random puzzle from the list of puzzles with the chosen difficulty level
#         puzzle = random.choice(
#             [p for p in puzzles if p["difficulty"] == difficulty])
#         # Shuffle the numbers in the puzzle
#         new_puzzle = [0 if x == 0 else random.choice(
#             [n for n in range(1, 5) if n != x]) for x in puzzle["solution"]]
#         # Add the new puzzle to the list of puzzles
#         puzzles.append({
#             "puzzle": new_puzzle,
#             "solution": puzzle["solution"],
#             "difficulty": difficulty
#         })


def print_board(sudoku):
    cell_id = 0
    m = int(np.cbrt(len(sudoku)))
    solution = np.ones(m*m, dtype=int)
    for i in range(m*m):
        for j in range(m):
            if(sudoku[i*m+j] == 1):
                solution[i] = j+1
                print(j+1, end='   ')
            if (j+1) % m == 0:
                print('|', end=' ')
        if (i+1) % m == 0:
            print()
            print('- ' * (m*m - m))
            cell_id = cell_id + 1
    return solution


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


def testingPuzzleGenerator(num_of_zeros):
    print(f"Puzzle {i+1}:")
    solution = genPuzzle()
    puzzle = makePuzzle(solution, num_of_zeros)

    for row in puzzle:
        print(row)
    print()
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
    print(solution)
    return solution


def get_hamming_distance(a, b):
    matrix1 = np.array(a).ravel()
    matrix2 = np.array(b).ravel()

    hamming_distance = sum(
        [1 if x != y else 0 for x, y in zip(matrix1, matrix2)])
    return hamming_distance


# generate a set of sudoku puzzles
num_of_zeros = 8
num_puzzles = 5
puzzles = []
solutions = []
for i in range(num_puzzles):
    puzzle, solution = testingPuzzleGenerator(num_of_zeros)
    puzzles.append(puzzle)
    solutions.append(solution)

# solve each puzzle and calculate Hamming distances and confusion matrices
# solve each puzzle and calculate Hamming distances and confusion matrices
hamming_dists = np.zeros((num_puzzles, num_puzzles))
confusion_mats = np.zeros((num_puzzles, num_puzzles))
for i in range(num_puzzles):
    for j in range(i, num_puzzles):
        if i == j:
            hamming_dists[i, j] = 0
            confusion_mats[i, j] = 0
        else:
            # solve puzzle i
            matW_i, vecT_i = hnetInitParameters(puzzles[i])
            n3 = puzzles[i].shape[0]**3
            vecS_i = rnd.binomial(n=1, p=0.05, size=n3)*2-1
            sol_i = hnetSimAnn(vecS_i, matW_i, vecT_i)
            print_board(sol_i)
            sol_i = decode(sol_i)

            # solve puzzle j
            matW_j, vecT_j = hnetInitParameters(puzzles[j])
            n3 = puzzles[j].shape[0]**3
            vecS_j = rnd.binomial(n=1, p=0.05, size=n3)*2-1
            sol_j = hnetSimAnn(vecS_j, matW_j, vecT_j)
            print_board(sol_j)
            sol_j = decode(sol_j)

            # calculate hamming distance and confusion matrix
            hamming_dists[i, j] = get_hamming_distance(sol_i, solutions[i])
            hamming_dists[j, i] = get_hamming_distance(sol_j, solutions[j])
            # confusion_mats[i, j] = np.sum(
            #     (puzzles[i] != 0) & (sol_j != solutions[j]))
            # confusion_mats[j, i] = np.sum(
            #     (puzzles[j] != 0) & (sol_i != solutions[i]))


# create a color-coded image for Hamming distances
plt.figure(figsize=(8, 8))
plt.imshow(hamming_dists, cmap='coolwarm', vmin=0, vmax=np.max(hamming_dists))
plt.colorbar()
plt.title('Hamming Distance Matrix')
plt.xlabel('Puzzle Index')
plt.ylabel('Puzzle Index')
plt.xticks(range(num_puzzles))
plt.yticks(range(num_puzzles))
plt.savefig('hamming_dist.png')
plt.show()

# create a color-coded image for confusion matrices
plt.figure(figsize=(8, 8))
plt.imshow(confusion_mats, cmap='Blues', vmin=0, vmax=np.max(confusion_mats))
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(range(num_puzzles))
plt.yticks(range(num_puzzles))
plt.savefig('confusion_mat.png')
plt.show()
