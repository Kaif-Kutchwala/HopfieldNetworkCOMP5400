import matplotlib.pyplot as plt
import json
import numpy as np
from tensorflow.keras.datasets import mnist


# Displays thresholded digits
def display_thresholded_digits():
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


# Calculates the average of every digit across all training samples
# in mnist dataset and stores in mnist_digits_average.json
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


# Returns average matrix for inputted list of matrices
def get_average_matrix(matrices):
    # Stack matrices along new axis to create 3D array
    # Return mean along first axis to get average matrix
    return np.mean(np.stack(matrices, axis=0), axis=0)


# Groups images by digits using corresponding labels
def group_by_digits(images, labels):
    # Store digits in dictionary
    digits = {}

    # Initialise empty lists for each digit
    for i in range(0, 10):
        digits[str(i)] = []

    # Use labels to group images
    for i in range(len(labels)):
        if str(labels[i]) in digits.keys():
            # reshape image to 784x1 dimensions
            im = np.reshape(images[i], (1, 784)).tolist()
            # append it to the corresponding label
            digits[str(labels[i])].append(im[0])

    # return dictionary
    return digits


# Applies threshold to all average digits and stores in
# mnist_digits_threshold.json
def store_thresholded_average_matrices():
    # Store output in dictionary
    output = {"data": []}

    # Get average matrices for all digits
    file = open("mnist_digits_average.json", "r")
    data = json.loads(file.read())
    file.close()

    # For each digit apply thresold and append to output
    for i in range(0, 10):
        average_matrix = np.array(data[str(i)])
        # 100 threshold determined through trial and error
        with_threshold = apply_threshold(average_matrix, 100)
        output["data"].append(with_threshold.tolist())

    # Store matrices with threshold applied in json
    file = open("mnist_digits_threshold.json", "w+")
    file.write(json.dumps(output))
    file.close()


# Applies a threshold to every element in inputted array
# Returns an array with 1s and -1s based on threshold
def apply_threshold(array, threshold):
    # If value is higher than threshold set to 1 else -1
    return np.where(array > threshold, 1, -1)

# Displays average digits


def display_average_digits():
    # Get average digits from files
    file = open("mnist_digits_average.json", "r")
    data = json.loads(file.read())
    file.close()

    # Plot average digits
    plt.figure(figsize=[10, 4])
    plt.title('Average Digits - MNIST')
    for id in range(len(data)):
        plt.subplot(1, 10, id + 1)
        plt.axis('off')
        plt.title(str(id))
        # Convert average matrix to np.array
        pattern = np.array(data[str(id)])
        # reshape pattern to 28x28 list and show on plot
        plt.imshow(pattern.reshape((28, 28)), cmap='gray')

    # Show plot
    plt.show()

# Displays hamming distance between thresholded digits on a 10x10 matrix


def display_hamming_distance():
    # get hamming distance between digits
    hamming_distances = get_digits_hamming()

    # Plot hamming distances
    plt.figure(figsize=[10, 4])
    plt.title('Hamming distances')

    # Add a label for every cell
    for y in range(hamming_distances.shape[0]):
        for x in range(hamming_distances.shape[1]):
            plt.text(x, y, '%.0f' % hamming_distances[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color="white"
                     )
    # Show hamming distances matrix
    plt.imshow(hamming_distances.reshape((10, 10)), cmap='gray')
    plt.show()

# Returns a list of hamming distances for each digit
# Every element at index 'i' is a list of the hamming distance of
# digit 'i' from every other digit by index


def get_digits_hamming():
    # Get thresholded digits
    file = open("mnist_digits_threshold.json", "r")
    digits = json.loads(file.read())["data"]
    file.close()

    # Initialise hamming distances to zero
    hamming_distances = np.zeros((10, 10))

    # For every digit
    for i in range(10):
        # Store hamming distance to every other digit
        for j in range(10):
            hamming_distances[i][j] = get_hamming_distance(
                digits[i], digits[j])

    # return hamming distances
    return hamming_distances

# Returns hamming distance between two matrices/arrays


def get_hamming_distance(a, b):
    # Flatten both matrices
    matrix1 = np.array(a).ravel()
    matrix2 = np.array(b).ravel()

    # Hamming distance is the count of every differing element
    hamming_distance = sum(
        [1 if x != y else 0 for x, y in zip(matrix1, matrix2)])

    # return hamming distance
    return hamming_distance

# Prints digits with their closest digits in ascending order of hamming distance


def get_closest_digits():
    # Get hamming distances for every digit
    hamming_distances = get_digits_hamming()

    # Loop through hamming distances for each digit
    for digit, hamming in enumerate(hamming_distances):
        # store the closest digits in a dictionary
        closest_digits = {}

        # Sort the hamming distances in ascending order
        for i in sorted(hamming):
            # Add the digit and its hamming distance to the closest_digits
            closest_digits[str(list(hamming).index(i))] = i

        # Print out the digit and its closest digits in ascending order of hamming distance
        print("Digit:", digit, "Closest Values:", closest_digits)
