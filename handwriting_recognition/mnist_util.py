import matplotlib.pyplot as plt
import json
import numpy as np
from tensorflow.keras.datasets import mnist


def display_digits():
    # Open file and get digit patterns
    file = open("mnist_digits_threshold.json", "r")
    patterns = json.loads(file.read())["data"]
    file.close()

    # Display patterns
    plt.figure(figsize=[20, 8])
    for i in range(min(len(patterns), 10)):
        plt.subplot(2, 5, i + 1)
        plt.title(i)
        plt.axis('off')
        plt.imshow(np.array(patterns[i]).reshape((28, 28)), cmap='gray')
    plt.show()


def calculate_average_digit_matrices():
    # Get images and labels
    (images, labels), _ = mnist.load_data()
    # Group images by digits
    digits = group_by_digits(images, labels)
    
    # Get average  matrix for each digit
    averages = {}
    for i in range(0, 10):
        averages[str(i)] = get_average_matrix(digits[str(i)])

    # Store averages in json file
    file = open("mnist_digits_average.json", "w+")
    file.write(json.dumps(averages))
    file.close()


def get_average_matrix(matrices):
    # Stack matrices along new axis to create 3D array
    # Return mean along first axis to get average matrix
    return np.mean(np.stack(matrices, axis=0), axis=0)


def group_by_digits(images, labels):
    # Stores digits
    digits = {}

    # Initialise empty lists for each digit
    for i in range(0, 10):
        digits[str(i)] = []

    # Use labels group images
    for i in range(len(labels)):
        if str(labels[i]) in digits.keys():
            im = np.reshape(images[i], (1, 784)).tolist()
            digits[str(labels[i])].append(im[0])
    return digits


def store_thresholded_average_matrices():
    output = {"data": []}

    # Get average matrices for all digits
    file = open("mnist_digits_average.json", "r")
    data = json.loads(file.read())
    file.close()

    # For each digit apply thresold and append to output
    for i in range(0, 10):
        average_matrix = np.array(data[str(i)])
        with_threshold = apply_threshold(average_matrix)
        output["data"].append(with_threshold.tolist())

    # Store matrices with threshold applied in json
    file = open("mnist_digits_threshold.json", "w+")
    file.write(json.dumps(output))
    file.close()


def apply_threshold(array):
    # If value is higher than 100 set to 1 else -1
    return np.where(array > 100, 1, -1)
