import numpy as np
import pygame

class HopfieldNetwork:
    def __init__(self, neuron_count):
        # AKA 'N'
        self.neuron_count = neuron_count
        # All neurons start with -1 as state
        self.neuron_states = -1 * np.ones(self.neuron_count)
        # Weight matrix is initialised to be of size N*N with all weights set to 0
        self.weight_matrix = np.zeros((neuron_count, neuron_count))

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
                break
        return state
    
    # def construct_hebb_matrix(xi):
    #     n = xi.shape[0]
    #     if len(xi.shape) == 1:
    #         w = np.outer(xi, xi) / n  # p = 1
    #     elif len(xi.shape) == 2:
    #         w = np.einsum("ik,jk", xi, xi) / n  # p > 1
    #     else:
    #         raise ValueError("Unexpected shape of input pattern xi: {}".format(xi.shape))
    #     np.fill_diagonal(w, 0)  # set diagonal elements to zero
    #     return w
        
