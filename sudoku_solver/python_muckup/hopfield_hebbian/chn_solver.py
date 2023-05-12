import numpy as np
import math

MaxTemp = 6
MinTemp = 5
CoolingRate = 1


class CHNSudokuSolver:
    def __init__(self):
        self.N = 9
        self.num_neurons = self.N * self.N * self.N

        # Initialize the weight matrix
        self.weights = np.zeros((self.num_neurons, self.num_neurons))

        # Initialize the temperature parameter
        self.T = MaxTemp

        # Initialize the cooling schedule
        self.alpha = 0.99

class CHNSudokuSolver:
    def __init__(self):
        self.num_units = 729
        self.network = np.zeros((self.num_units, self.num_units))
        self.bias = np.zeros((self.num_units,))
        
    def train(self, puzzles):
        # Convert puzzles to binary matrices
        matrices = []
        for puzzle in puzzles:
            matrix = []
            for i in range(81):
                if puzzle[i] != ".":
                    row = [0] * 9 * i + [1] + [0] * (9 * (80 - i))
                else:
                    row = [1] * 9 * i + [0] + [1] * (9 * (80 - i))
                matrix.append(row)
            matrices.append(matrix)

        # Reshape matrices to 1D arrays
        inputs = []
        for matrix in matrices:
            inputs.append(np.concatenate(matrix).flatten())

        # Train network using simulated annealing
        for i in range(len(inputs)):
            input_pattern = inputs[i]
            target_pattern = input_pattern.copy()
            self.network, self.bias = simulated_annealing(
                self.network,
                self.bias,
                input_pattern,
                target_pattern,
                num_iterations=1000,
                temperature_schedule="linear",
                initial_temperature=100,
                final_temperature=0.1,
                cooling_rate=0.99,
                random_seed=None,
            )

    def solve_sudoku(self, puzzle):
        # Convert puzzle to binary matrix
        matrix = []
        for i in range(81):
            if puzzle[i] != ".":
                row = [0] * 9 * i + [1] + [0] * (9 * (80 - i))
            else:
                row = [1] * 9 * i + [0] + [1] * (9 * (80 - i))
            matrix.append(row)
        input_pattern = np.concatenate(matrix).flatten()

        # Solve Sudoku using CHN and simulated annealing
        output_pattern = self._chn_solve(input_pattern)

        # Convert output pattern to Sudoku grid
        grid = self._vector_to_sudoku(output_pattern)

        return grid


    def sudoku_to_vector(self, grid):
        # Convert a Sudoku grid to a vector of binary values
        vec = np.zeros(self.num_neurons)
        for i in range(self.N):
            for j in range(self.N):
                if grid[i][j] != 0:
                    index = (i * self.N + j) * self.N + int(grid[i][j]) - 1
                    vec[index] = 1
                else:
                    subarray = vec[(i * self.N + j) *
                                   self.N: (i * self.N + j + 1) * self.N]
                    index = np.where(subarray == 0)[0]
                    vec[(i * self.N + j) * self.N + index] = 1
        return vec

    def vector_to_sudoku(self, vec):
        # Convert a vector of binary values to a Sudoku grid
        grid = np.zeros((self.N, self.N), dtype=int)
        for i in range(self.N):
            for j in range(self.N):
                subarray = vec[(i * self.N + j) *
                               self.N: (i * self.N + j + 1) * self.N]
                index = np.where(subarray == 1)[0]
                if len(index) == 1:
                    grid[i][j] = index[0] + 1
        return grid

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1.0 / (1.0 + np.exp(-x))

    def update_weights(self, vecs):
        # Update the weight matrix using Hebbian learning
        for vec in vecs:
            for i in range(self.num_neurons):
                for j in range(i, self.num_neurons):
                    if i == j:
                        self.weights[i][j] = 0
                    else:
                        self.weights[i][j] += (vec[i] * vec[j] - 0.5) / self.num_neurons
                        self.weights[j][i] = self.weights[i][j]

    def recall(self, vec):
        # Perform a recall iteration using the CHN model
        prev_output = vec
        new_output = self.sigmoid(np.dot(self.weights, prev_output))
        while not np.array_equal(prev_output, new_output):
            prev_output = new_output
            new_output = self.sigmoid(np.dot(self.weights, prev_output))
        return new_output

    def solve_sudoku(self, grid):
        # Solve the Sudoku puzzle using the CHN solver with simulated annealing
        vec = self.sudoku_to_vector(grid)

        # Initialize the solution vector with the input vector
        solution = vec

        # Set the initial energy and temperature
        energy = np.dot(solution, np.dot(self.weights, solution)) / -2.0
        T = self.T

        # Loop until the temperature reaches a minimum value
        while T > MinTemp:
            # Choose a random neuron and flip its state
            index = np.random.randint(self.num_neurons)
            new_solution = np.copy(solution)
            new_solution[index] = 1 - solution[index]

            # Calculate the new energy
            new_energy = np.dot(new_solution, np.dot(
                self.weights, new_solution)) / -2.0

            # Calculate the energy difference
            delta_E = new_energy - energy

            # Decide whether to accept the new state
            if delta_E < 0 or math.exp(-delta_E / T) > np.random.uniform():
                solution = new_solution
                energy = new_energy

            # Update the temperature
            T *= self.alpha

        # Convert the solution vector to a Sudoku grid and return it
        return self.vector_to_sudoku(solution)

# def vector_to_sudoku(self, vec):
#     # Convert a vector of binary values to a Sudoku grid
#     grid = np.zeros((self.N, self.N), dtype=int)
#     for i in range(self.N):
#         for j in range(self.N):
#             subarray = vec[(i * self.N + j) * self.N : (i * self.N + j + 1) * self.N]
#             index = np.where(subarray == 1)[0]
#             if len(index) == 1:
#                 grid[i][j] = index[0] + 1
#     return grid

#     def sudoku_to_vector(self, grid):
#         # Convert a Sudoku grid to a vector of binary values
#         vec = np.zeros(self.num_neurons)
#         for i in range(self.N):
#             for j in range(self.N):
#                 if grid[i][j] != 0:
#                     vec[(i * self.N + j) * self.N + int(grid[i][j]) - 1] = 1
#                 else:
#                     for k in range(self.N):
#                         vec[(i * self.N + j) * self.N + k] = 1
#         return vec

#     def solve_sudoku(self, grid):
#         # Solve a Sudoku puzzle using the continuous Hopfield network
#         input = self.sudoku_to_vector(grid)
#         output = self.recall(input)
#         return self.vector_to_sudoku(output)
