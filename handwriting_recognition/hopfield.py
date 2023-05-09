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

    def train(self, memories, learning_rule="pi"):
        if learning_rule == "pi":
            # Pseudo Inverse
            memory_count = len(memories)
            c = np.tensordot(memories, memories, axes=(
                (1), (1))) / memory_count
            cinv = np.linalg.inv(c)
            for k, l in product(range(memory_count), range(memory_count)):
                self.weights = self.weights + \
                    cinv[k, l] * memories[k] * memories[l].reshape((-1, 1))
            self.weights = self.weights / memory_count
            np.fill_diagonal(self.weights, 0)
        elif learning_rule == "hebb":
            for pattern in memories:
                self.weights += np.outer(pattern, pattern)
            np.fill_diagonal(self.weights, 0)
        elif learning_rule == "storkey":
            # Iterate through each pattern mu
            for mu in range(len(memories)):
                print("Currently on Pattern", mu + 1, "of", len(memories))
                # Set each weight between neuron i and j
                for i in range(self.n):
                    for j in range((i - self.n) % self.n):
                        # Calculate local field
                        local_field = 0
                        for k in range(self.n):
                            if (k != i & k != j):
                                if not np.isnan(self.weights[i][k]):
                                    local_field += self.weights[i][k] * \
                                        memories[mu][k]
                        # Increment weight for the pattern
                        self.weights[i][j] += (1/self.n) * ((memories[mu][i] * memories[mu][j]) -
                                                            ((1/self.n) * memories[mu][i] * local_field) -
                                                            ((1/self.n) * local_field * memories[mu][j]))
            np.fill_diagonal(self.weights, 0)
        else:
            raise ValueError("Learning rule not supported. Options are 'pi', 'hebb' and 'storkey'.")

    def recall(self, input_pattern, iterations):
        order = np.arange(0, self.n)
        for _ in range(iterations):
            np.random.shuffle(order)
            state = np.copy(input_pattern)
            no_change_count = 0
            for i in order:
                weights = self.weights[i, :]
                state[i] = self.sign(np.dot(weights, state))
            if np.array_equal(state, input_pattern):
                no_change_count += 1
                if no_change_count > 3:
                    break
            else:
                no_change_count = 0
        return state, self.get_result_label(state)
    def sign(self, value):
        return 1 if value >= 0 else -1

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
