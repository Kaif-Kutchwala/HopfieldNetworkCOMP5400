import numpy as np
import json
from itertools import product


class HopfieldNetwork:
    def __init__(self, neuron_count):
        # AKA 'N'
        self.n = neuron_count
        # All neurons start with -1 as state
        self.neuron_states = -1 * np.ones(self.n)
        # Weight matrix is initialised to be of size N*N with all weights set to 0
        self.weights = np.zeros((self.n, self.n))
        self.order = np.arange(0, self.n)

    def train(self, patterns, learning="pi"):
        if learning == "pi":
            # Pseudo Inverse
            num_of_patterns = len(patterns)
            c = np.tensordot(patterns, patterns, axes=(
                (1), (1))) / num_of_patterns
            cinv = np.linalg.inv(c)
            for k, l in product(range(num_of_patterns), range(num_of_patterns)):
                self.weights = self.weights + \
                    cinv[k, l] * patterns[k] * patterns[l].reshape((-1, 1))
            self.weights = self.weights / num_of_patterns
        elif learning == "hebb":
            for pattern in patterns:
                self.weights += np.outer(pattern, pattern)
            np.fill_diagonal(self.weights, 0)
        elif learning == "storkey":
            # Iterate through each pattern mu
            for mu in range(len(patterns)):
                print("Currently on Pattern", mu + 1, "of", len(patterns))
                # Set each weight between neuron i and j
                for i in range(self.n):
                    for j in range((i - self.n) % self.n):
                        # Calculate local field
                        local_field = 0
                        for k in range(self.n):
                            if (k != i & k != j):
                                if not np.isnan(self.weights[i][k]):
                                    local_field += self.weights[i][k] * \
                                        patterns[mu][k]
                        # Increment weight for the pattern
                        self.weights[i][j] += (1/self.n) * ((patterns[mu][i] * patterns[mu][j]) -
                                                            ((1/self.n) * patterns[mu][i] * local_field) -
                                                            ((1/self.n) * local_field * patterns[mu][j]))
            np.fill_diagonal(self.weights, 0)
        else:
            pass

    def recall(self, partial_pattern, iterations):
        for _ in range(iterations):
            np.random.shuffle(self.order)
            state = np.copy(partial_pattern)
            no_change_count = 0
            for i in self.order:
                weights = self.weights[i, :]
                state[i] = np.sign(np.dot(weights, state))
            if np.array_equal(state, partial_pattern):
                no_change_count += 1
                if no_change_count > 3:
                    break
        return state, self.get_result_label(state)

    def get_result_label(self, pattern):
        file = open("mnist_digits_threshold.json", "r")
        dataset = json.loads(file.read())["data"]
        file.close()

        closest_digit = 0
        lowest_hamming_distance = len(pattern)
        for label, digit in enumerate(dataset):
            hamming_distance = self.get_hamming_distance(pattern, digit)
            if hamming_distance < lowest_hamming_distance:
                closest_digit = label
                lowest_hamming_distance = hamming_distance

        return closest_digit

    def get_hamming_distance(self, a, b):
        matrix1 = np.array(a).ravel()
        matrix2 = np.array(b).ravel()

        hamming_distance = sum(
            [1 if x != y else 0 for x, y in zip(matrix1, matrix2)])
        return hamming_distance
