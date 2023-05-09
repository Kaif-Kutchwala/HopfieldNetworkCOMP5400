from hopfield import HopfieldNetwork
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import seaborn as sn
import pandas as pd
import random


def test_hopfield_network_with_mnist():
    # get network training dataset
    file = open("mnist_digits_threshold.json", "r")
    dataset = json.loads(file.read())["data"]
    file.close()

    # create Hopfield network with 784 neurons
    hn = HopfieldNetwork(784)

    # define patterns to train network
    patterns = np.array(dataset)

    # train the network on the patterns using pseudoinverse rule
    rule = "pi"
    hn.train(patterns, rule)

    # load the mnist test dataset
    _, (x_test, y_test) = mnist.load_data()
    # create variable to store test results
    tests_passed = 0
    num_of_tests = 100
    confusion_matrix = initialize_confusion_matrix() # create empty confusion matrix
    
    # define iterations to simulate network for each test
    iteration_count = 5

    # loop over every test sample
    for i in range(num_of_tests):
        # reshape test sample to single list of 784 elements for network
        input_pattern = np.where(np.array(x_test[i]).reshape(784, 1) > 100, 1, -1)
        # store true label
        label = y_test[i]

        # retrieve the closest stored pattern
        output_pattern, prediction = hn.recall(input_pattern, iteration_count)

        # if true label matches prediction increment tests passed
        if int(label) == int(prediction): tests_passed += 1
        # Add test result to confusion matrix
        confusion_matrix = add_test_result(confusion_matrix, label, prediction)

        # print test result
        print("Test", i + 1, "Input:", label, "Prediction:", prediction)

    # Calculate success rate using number of tests passed
    success_rate = tests_passed / num_of_tests

    # Store results
    results = {}
    results["Iterations"] = iteration_count
    results["Tests Passed"] = tests_passed
    results["Total Number of Tests"] = num_of_tests
    results["Success Rate"] = success_rate
    results["Confusion Matrix"] = confusion_matrix

    # Write results to json file
    file = open("results_" + str(rule) + "_" + str(iteration_count) + ".json", "w+")
    file.write(json.dumps(results))
    file.close()

    # print test results
    print("Tests passed:", tests_passed)
    print("Total tests:", num_of_tests)
    print("Success Rate:", success_rate)

    # display the confusion matrix
    display_confusion_matrix(confusion_matrix)

# Returns an object matching the required format for the confusion matrix
def initialize_confusion_matrix():
    results = {}
    for i in range(10):
        results[str(i)] = []
        results[str(i) + "-testcount"] = 0
        for j in range(10):
            results[str(i)].append(0)
    return results

# Adds a test result to the confusion matrix
def add_test_result(confusion_matrix, label, prediction):
    # Increment value for predicted label under true label
    confusion_matrix[str(label)][int(prediction)] += 1
    # Increment test count for true label
    confusion_matrix[str(label)+"-testcount"] +=1
    # return modified confusion matrix
    return confusion_matrix

# Data should be an object with keys for class labels i.e. "{digit}" 
# and the corresponding total samples for each class label i.e. "{digit}-testcount" 
# for every digit i.e. 0-9.
# The value for the class label should be a list of coressponding predicted labels
# The value for the test count should be the total number of samples with the true label in the dataset
# See example of confusion matrix in confusion_matrix.json
def display_confusion_matrix(data):
    confusion_matrix = []
    for i in range(10):
        confusion_matrix.append(data[str(i)])

    df_cm = pd.DataFrame(np.array(confusion_matrix), index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
    plt.figure(figsize = (15,15))
    sn.heatmap(df_cm, annot=True, cmap='gray_r', linecolor="black", linewidths=1, fmt='.0f', annot_kws={"size": 20})
    # plt.imshow( (np.ones((10, 10)) - confusion_matrix), cmap='gray', origin="lower" )
    plt.show()

# Selects a random test digit from the mnist dataset and displays
# the state of the network as it progresses to its final state.
# Number of iterations of the network and the steps at which to record
# the state of the network can be modified as per requirements.
def show_classification_progession():
    # get network training dataset
    file = open("mnist_digits_threshold.json", "r")
    dataset = json.loads(file.read())["data"]
    file.close()

    # create Hopfield network with 784 neurons
    hn = HopfieldNetwork(784)

    # define patterns to train network
    patterns = np.array(dataset)

    # train the network on the patterns
    hn.train(patterns, "pi")

    # load mnist test data
    _, (x_test, y_test) = mnist.load_data()

    # pick a random test to use for run
    number_of_test_patterns = len(x_test) 
    random_test_id = random.randint(0, number_of_test_patterns - 1)

    # reshape mnist pattern to single list with 784 elements
    input_pattern = np.where(np.array(x_test[random_test_id]).reshape(784, 1) > 100, 1, -1)

    # Define steps to record for each iteration
    steps_to_record = [0, 100, 400, 500, 750]
    # Define number of iterations
    iteration_count = 2
    # Simulate hopfield network, but us recall_with_steps to get state at all required steps
    output_pattern_states, predictions = hn.recall_with_steps(input_pattern, steps_to_record, iteration_count)

    # Plot the results
    plt.figure(figsize=[10, 4])
    # For each pattern in output
    for iteration, pattern in enumerate(output_pattern_states):
        # Create a subplot
        plt.subplot(1, len(steps_to_record*iteration_count), iteration + 1)
        # Turn axis labels off
        plt.axis('off')
        # Set title to be network's prediction for state
        plt.title(str(predictions[iteration]))
        # Display state in subplot
        plt.imshow(pattern.reshape((28, 28)), cmap='gray')
    # Show the plot
    plt.show()

# Displays a set of random samples and the network's prediction as well as final state
# Use test_count to specify number of tests to conduct and display
# Top row -> Input patterns with true labels, Bottom row -> output patterns with predicted labels
def show_random_tests(test_count):
    file = open("mnist_digits_threshold.json", "r")
    dataset = json.loads(file.read())["data"]
    file.close()

    # create Hopfield network with 784 neurons
    hn = HopfieldNetwork(784)

    # define patterns to train network
    patterns = np.array(dataset)

    # train the network on the patterns
    hn.train(patterns, "pi")

    # create variables to store input patterns, output patterns and labels
    inputs = []
    input_labels = []
    output_labels = []
    outputs = []

    # load mnist test dataset
    _, (x_test, y_test) = mnist.load_data()

    # create a list of specified random test samples
    random_test_samples = [random.randint(0, len(x_test) - 1) for _ in range(test_count)]

    # For every test sample
    for test_sample_index in random_test_samples:
        # reshape the mnist sample to match dimensions 784x1
        input_pattern = np.where(np.array(x_test[test_sample_index]).reshape(784, 1) > 100, 1, -1)
        # store input pattern and label
        inputs.append(input_pattern)
        input_labels.append(y_test[test_sample_index])

        # retrieve the closest stored pattern
        output_pattern, prediction = hn.recall(input_pattern, 100)
        # store output pattern and predicted label
        output_labels.append(prediction)
        outputs.append(output_pattern)

    # Plot patterns
    plt.figure(figsize=[10, 4])
    # For every test sample
    for x in range(len(random_test_samples)):
        # Plot the input pattern with true label
        plt.subplot(2, len(random_test_samples), x + 1)
        plt.axis('off')
        plt.title(str(input_labels[x]))
        plt.imshow(inputs[x].reshape((28, 28)), cmap='gray')
        
        # Plot the output pattern with predicted label below it
        plt.subplot(2, len(random_test_samples), x + len(random_test_samples) + 1)
        plt.axis('off')
        plt.title(str(output_labels[x]))
        plt.imshow(outputs[x].reshape((28, 28)), cmap='gray')
    
    # Show the plot
    plt.show()