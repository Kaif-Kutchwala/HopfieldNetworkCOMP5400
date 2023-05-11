import numpy as np
import numpy.random as rnd


def hnetInitParameters(sudoku, lH=4., lG=2., lR=1., lC=1., lB=1.):
    n1 = sudoku . shape[0]
    n2 = n1 ** 2
    n3 = n1 ** 3
    m = np . sqrt(n1). astype(int)
    nh = np .sum(sudoku > 0)
    # number of hints
    vec1_m = np . ones(m)
    vec1_n1 = np . ones(n1)
    vec1_n2 = np . ones(n2)
    vec1_n3 = np . ones(n3)
    vec1_nh = np . ones(nh)
    matI_m = np . eye(m)
    matI_n1 = np . eye(n1)
    matI_n2 = np . eye(n2)
    # matrix for hint constraints
    h = 0
    matH = np . zeros((nh, n3))
    for i in range(n1):
        for j in range(n1):
            v = sudoku[i, j]
            if v > 0:
                matH[h, i * n2 + j * n1: i * n2 + j * n1 + n1] = matI_n1[v - 1]
                h = h + 1
    # matrices for cell , row , column , and block constraints
    matG = np . kron(matI_n2, vec1_n1)
    matR = np . kron(vec1_n1, matI_n1)
    matR = np . kron(matI_n1, matR)
    matC = np . kron(vec1_n1, matI_n2)
    matB = np . kron(vec1_m, matI_n1)
    matB = np . kron(matI_m, matB)
    matB = np . kron(vec1_m, matB)
    matB = np . kron(matI_m, matB)
    # matrix and vector for binary QUBO
    matP = lH * matH . T @ matH
    matP += lG * matG . T @ matG
    matP += lR * matR . T @ matR
    matP += lC * matC . T @ matC
    matP += lB * matB . T @ matB
    vecP = lH * 2 * vec1_nh @ matH
    vecP += lG * 2 * vec1_n2 @ matG
    vecP += lR * 2 * vec1_n2 @ matR
    vecP += lC * 2 * vec1_n2 @ matC
    vecP += lB * 2 * vec1_n2 @ matB
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


sudoku = np . array([[0, 0, 2, 0],
                     [3, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 2, 0, 0]])

matW, vecT = hnetInitParameters(sudoku)

n3 = sudoku.shape[0]**3
vecS = rnd.binomial(n=1, p=0.05, size=n3)*2-1

vecS = hnetSimAnn(vecS, matW, vecT)

print_board(vecS)
decode(vecS)
