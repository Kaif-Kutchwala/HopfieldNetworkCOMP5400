import numpy as np
from hopfield import HopfieldNetwork

# Load training data
training_data = np.load("sudoku_training_data.npy")

# Create Hopfield network with 729 neurons (81 cells x 9 possible digits)
hopfield_network = HopfieldNetwork(729)

# Train Hopfield network using the pseudoinverse learning rule
patterns = training_data.reshape(-1, 729)
W = np.linalg.pinv(patterns.T @ patterns - len(patterns) * np.eye(729))
hopfield_network.weights = W
