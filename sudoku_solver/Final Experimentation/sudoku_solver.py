import numpy as np
import copy
import random


class CHNSudokuSolver:
    def __init__(self):
        self.W = None

    def one_hot_encoding(self, x):
        """
        One-hot encodes an input vector x of length n into an n x n matrix
        """
        n = len(x)
        encoded = np.zeros((n, n))
        for i in range(n):
            if x[i] != 0:
                encoded[i][x[i]-1] = 1
            else:
                encoded[i] = np.ones(n)
        return encoded

    def decode(self, x):
        """
        Decodes a one-hot encoded vector x of length n into an array of integers
        """
        decoded = []
        for i in range(len(x)):
            decoded.append(np.argmax(x[i])+1)
        return decoded

    def sigmoid(self, x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def energy(self, state):
        """
        Calculates the energy of a state
        """
        energy = 0
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j] == 0:
                    continue
                for k in range(len(state)):
                    if k != j:
                        energy -= self.W[i][j][k] * state[i][j] * state[i][k]
                    if k != i:
                        energy -= self.W[i][j][k] * state[i][j] * state[k][j]
        return energy

    def simulate_annealing(self, state):
        """
        Simulated annealing for optimizing state
        """
        temperature = 10
        cooling_rate = 0.999
        min_temperature = 1e-3
        current_energy = self.energy(state)
        while temperature > min_temperature:
            new_state = copy.deepcopy(state)
            row = random.randint(0, 8)
            col = random.randint(0, 8)
            while new_state[row][col] != 0:
                row = random.randint(0, 8)
                col = random.randint(0, 8)
            new_state[row][col] = 1 - new_state[row][col]
            new_energy = self.energy(new_state)
            delta_energy = new_energy - current_energy
            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
                state = copy.deepcopy(new_state)
                current_energy = new_energy
            temperature *= cooling_rate
        return state

    def energy(self, state):
        """
        Calculates the energy of a state
        """
        energy = 0
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j].item() == 0:
                    continue
                for k in range(len(state)):
                    if k != j:
                        energy -= self.W[i][j][k] * state[i][j] * state[i][k]
                    if k != i:
                        energy -= self.W[i][j][k] * state[i][j] * state[k][j]

                # Calculate contribution from neighboring cells in the same row
                row_neighbors = [state[i][x]
                                 for x in range(len(state)) if x != j]
                energy -= np.dot(row_neighbors, self.W[i][j][:]) * state[i][j]

                # Calculate contribution from neighboring cells in the same column
                col_neighbors = [state[x][j]
                                 for x in range(len(state)) if x != i]
                energy -= np.dot(col_neighbors, self.W[i][:, j]) * state[i][j]

        return energy

    def train(self, puzzles, epochs=10, verbose=True):
        """
        Trains the weights of the CHN for solving Sudoku puzzles
        """
        # Convert puzzles to one-hot encoded format
        encoded_puzzles = np.array(
            [np.array([self.one_hot_encoding(row) for row in puzzle]) for puzzle in puzzles])

        # Initialize weights
        self.W = np.zeros((81, 9, 9))
        for i in range(81):
            for j in range(9):
                for k in range(9):
                    if j == k:
                        continue
                    self.W[i][j][k] = 1

        # Train weights
        for epoch in range(epochs):
            total_energy = 0
            for puzzle in encoded_puzzles:
                # Initialize state with puzzle
                state = copy.deepcopy(puzzle)

                # Simulated annealing
                state = self.simulate_annealing(state)

                # Calculate energy of final state
                energy = self.energy(state)
                total_energy += energy

                # Update weights
                for i in range(len(state)):
                    for j in range(len(state)):
                        if state[i][j] == 0:
                            continue
                        for k in range(len(state)):
                            if k != j:
                                self.W[i][j][k] += state[i][j] * \
                                    state[i][k] / 2
                            if k != i:
                                self.W[i][j][k] += state[i][j] * \
                                    state[k][j] / 2

            if verbose:
                print("Epoch {}: total energy = {}".format(epoch+1, total_energy))

        if verbose:
            print("Training completed.")

    def solve_sudoku(self, puzzle):
        # One-hot encoding of puzzle
        encoded_puzzle = np.array(
            [self.one_hot_encoding(row) for row in puzzle])

        # Initialize weights
        self.W = np.zeros((81, 9, 9))
        for i in range(81):
            for j in range(9):
                for k in range(9):
                    if j == k:
                        continue
                    self.W[i][j][k] = 1

        # Train weights
        for i in range(len(encoded_puzzle)):
            for j in range(len(encoded_puzzle)):
                if encoded_puzzle[i][j].sum() == 1:
                    k = np.argmax(encoded_puzzle[i][j])
                    for l in range(len(encoded_puzzle)):
                        if l == i:
                            continue
                        self.W[l][j][k] -= 1

        # Simulated annealing for optimizing state
        optimized_state = self.simulate_annealing(encoded_puzzle)

        # Decode optimized state into a solution
        decoded_solution = np.array([self.decode(row)
                                    for row in optimized_state])

        return decoded_solution


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

solver = CHNSudokuSolver()
solution = solver.solve_sudoku(grid)
print(solution)
