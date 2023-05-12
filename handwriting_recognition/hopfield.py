import numpy as np
import json
from itertools import product


class HopfieldNetwork:
    def __init__(self, neuron_count):
        # AKA 'N'
        self.n = neuron_count
        # Weight matrix is initialised to be of size N*N with all weights set to 0
        self.weights = np.zeros((self.n, self.n))

    def train(self, memories, learning_rule="pi"):
        if learning_rule == "pi":
            # Pseudo Inverse
            # Count the number of memories
            memory_count = len(memories)
            # Calculate 'C' term
            c = np.tensordot(memories, memories, axes=(
                (1), (1))) / memory_count
            # Calculate inverse of C
            cinv = np.linalg.inv(c)
            # Apply PI equation
            for k, l in product(range(memory_count), range(memory_count)):
                self.weights = self.weights + \
                    cinv[k, l] * memories[k] * memories[l].reshape((-1, 1))
            self.weights = self.weights / memory_count
            # Fill diagonal with zeroes
            np.fill_diagonal(self.weights, 0)
        elif learning_rule == "hebb":
            # Use outer to apply hebb equation
            for pattern in memories:
                self.weights += np.outer(pattern, pattern)
            # Fill diagonal with zeroes
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
            # Fill diagonal with zeroes
            np.fill_diagonal(self.weights, 0)
        else:
            raise ValueError("Learning rule not supported. Options are 'pi', 'hebb' and 'storkey'.")

    def recall(self, input_pattern, iterations):
        # Get order of neurons to update (not shuffled)
        order = np.arange(0, self.n)
        # Perform update 'iterations' amount of times
        for _ in range(iterations):
            # Shuffle the order to randomly update neurons
            np.random.shuffle(order)
            # create a copy the state
            state = np.copy(input_pattern)
            # counts the number of updates that resulted in no changes
            no_change_count = 0
            for i in order:
                # For every neuron in the order, update the state
                weights = self.weights[i, :]
                state[i] = self.sign(np.dot(weights, state))
            # if no change occured
            if np.array_equal(state, input_pattern):
                # increment no change count
                no_change_count += 1
                # if no changes occured over 20 consecutive neuron updates, break
                if no_change_count > 20:
                    break
            else:
                # else reset no change count
                no_change_count = 0

        # Return final state and prediction
        return state, self.get_result_label(state)
    
    # Same as recall except state of the network is stored at multiple intervals
    # and all states are returned along with predictions for each state
    def recall_with_steps(self, input_pattern, steps_to_record, iteration_count=1):
        network_states = []
        predictions = []
        order = np.arange(0, self.n)
        np.random.shuffle(order)
        state = np.copy(input_pattern)
        no_change_count = 0
        for _ in range(iteration_count):
            for iteration, i in enumerate(order):
                if iteration in steps_to_record:
                    network_states.append(np.copy(state))
                    predictions.append(self.get_result_label(np.copy(state)))
                weights = self.weights[i, :]
                state[i] = self.sign(np.dot(weights, state))
            if np.array_equal(state, input_pattern):
                no_change_count += 1
                if no_change_count > 3:
                    break
            else:
                no_change_count = 0
        return network_states, predictions
    
    # Returns 1 if value is greater than zero else -1
    def sign(self, value):
        return 1 if value >= 0 else -1

    # Determines label for state by finding the closest digit using hamming distance
    def get_result_label(self, pattern):
        # Get memories
        file = open("mnist_digits_threshold.json", "r")
        dataset = json.loads(file.read())["data"]
        file.close()

        # start with 0
        closest_digit = 0
        # lowest hamming distance is 784 i.e. maximum possible hamming distance
        lowest_hamming_distance = len(pattern)
        # iterate over all digits
        for label, digit in enumerate(dataset):
            # calculate hamming distance
            hamming_distance = self.get_hamming_distance(pattern, digit)
            # if it is the lowest update the closest digit lowest hamming distance
            if hamming_distance < lowest_hamming_distance:
                closest_digit = label
                lowest_hamming_distance = hamming_distance

        # return the label of the closest digit
        return closest_digit

    # Returns the hamming distance between two arrays
    def get_hamming_distance(self, a, b):
        # Flatten arrays
        matrix1 = np.array(a).ravel()
        matrix2 = np.array(b).ravel()

        # sum number of differences for hamming distance
        hamming_distance = sum(
            [1 if x != y else 0 for x, y in zip(matrix1, matrix2)])
        
        return hamming_distance
