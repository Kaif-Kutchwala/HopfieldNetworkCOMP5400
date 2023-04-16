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

        hamming_distance = sum([1 if x != y else 0 for x,y in zip(matrix1,matrix2)])
        return hamming_distance