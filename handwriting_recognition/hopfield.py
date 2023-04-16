import numpy as np
import json

class HopfieldNetwork:
    def __init__(self, neuron_count):
        # AKA 'N'
        self.n = neuron_count
        # All neurons start with -1 as state
        self.neuron_states = -1 * np.ones(self.n)
        # Weight matrix is initialised to be of size N*N with all weights set to 0
        self.weights = np.zeros((self.n, self.n))
        self.order = np.arange(0, self.n)

    def update(self, input_pattern):
        activation = self.get_activation(input_pattern)
        return np.sign(activation)
    
    def get_activation(self, input_pattern):
        return np.dot(self.weight_matrix, input_pattern)

    def train(self, patterns):
        num_patterns = patterns.shape[0]
        for p in range(num_patterns):
            pattern = patterns[p, :]
            for i in range(self.neuron_count):
                for j in range(self.neuron_count):
                    if i != j:
                        self.weight_matrix[i, j] += pattern[i] * pattern[j]
        self.weight_matrix //= self.neuron_count

    def recall(self, partial_pattern, iterations=10):
        """
        Recall a stored pattern closest to the input pattern.
        """
        state = np.copy(partial_pattern)
        for _ in range(iterations):
            for i in range(self.neuron_count):
                weights = self.weight_matrix[i, :]
                state[i] = np.sign(np.dot(weights, state))
            if np.array_equal(state, partial_pattern):
    def get_hamming_distance(self, a, b):
        matrix1 = np.array(a).ravel()
        matrix2 = np.array(b).ravel()

        hamming_distance = sum([1 if x != y else 0 for x,y in zip(matrix1,matrix2)])
        return hamming_distance