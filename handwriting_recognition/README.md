# Hopfield Networks: Handwriting Recognition using the MNIST Database

This repository contains an implementation of a Hopfield Network, a script to process digits in the MNIST database and a pygame script that allows you to draw digits and classify them using the Hopfield Network.

```
Before running any script make sure all the dependencies are installed.
Errors should tell you which dependencies are missing.
```

## Simple Usage of Hopfield Network Implementation
```python
from hopfield import HopfieldNetwork
import numpy as np
import matplotlib.pyplot as plt

# create Hopfield network with 784 neurons
hn = HopfieldNetwork(784)

# define patterns to train network
patterns = np.array(dataset)

# train the network on the patterns
hn.train(patterns, "pi")

# retrieve the closest stored pattern
output_pattern, label = hn.recall(input_pattern, 1000)
```

For this to work however, the `mnist_digits_threshold.json` must exist. See usage for other files below to find out how to generate this file.

#### Test scripts can be found in main that use the MNIST database to run various tests, see usage for main.py to learn more.


The following files are provided:

| File | Purpose |
| ---  | --- |
| `hopfield.py`      | Python Implementation of Hopfield Network |
| `mnist_util.py`     | Contains methods that can be used to process digits in the MNIST dataset. |
| `draw_and_classify.py`        | A pygame application that allows user to draw a digit and classify it using the Hopfield Network |
| `main.py` | Contains test scripts used for testing the Hopfield network. |
| `confusion_matrix.json` | Contains an example of the confusion matrix generated for this project. |
| `mnist_digits_average.json` | Contains the average 28x28 image for every digit in the MNIST dataset.|
| `mnist_digits_threshold.json` | Contains the thresholded 28x28 images for every digit in the mnist_digits_average.json.|

# Usage: `main.py`

In `main.py` you will find four important methods
## 1. test_hopfield_network_with_mnist
This uses the HN implementation to classify the test dataset of the MNIST dataset and stores the results in a file as well as displays the confusion matrix using matplotlib.

The learning rule can be modified by changing the value of `rule` between `hebb`, `pi` or `storkey`.

Simply add `test_hopfield_network_with_mnist()` at the end of `main.py` and in your terminal execute `python main.py` to run the tests.

## 2. show_classification_progession
This uses the HN implementation to classify a random test sample in the MNIST dataset and displays the progression of the network state using matplotlib.

The steps at which to record the state of the network can be changed by modifying the `steps_to_record` list and the number of iterations can be changed by modifying the `iteration_count` variable.

Simply add `show_classification_progession()` at the end of `main.py` and in your terminal execute `python main.py` to run the test.

## 3. show_random_tests
This uses the HN implementation to classify a set of random test samples picked from the MNIST dataset and displays the classification results using matplotlib.

The number of random test samples selected can be set using the `test_count` function argument.

Simply add `show_random_tests(10)` at the end of `main.py` and in your terminal execute `python main.py` to run the test for 10 random samples.

## 4. classify
This uses the HN implementation to classify the inputted pattern.

The input pattern must be passed as a function argument and must have the dimensions of `784x1` where every element is either 1 or -1.

Simply add `classify({input_pattern_here})` at the end of `main.py` and in your terminal execute `python main.py` to run classify the input pattern.

---
---
# Usage: `mnist_util.py`
This file provides multiple methods that were used to process the MNIST dataset.

## `calculate_average_digit_matrices`
`calculate_average_digit_matrices` was used to calculate the average matrix for every digit across all 60,000 training samples for each digit. The results are stored in a file called `mnist_digits_average.json`. The results when displayed using `display_average_digits` shows the following results:

![Average digits](https://i.imgur.com/WxjM8iV.png)

## `store_thresholded_average_matrices`
Applies a threshold to the average digits and stores it in a file called `mnist_digits_threshold.json`. The results when displayed using `display_thresholded_digits` shows the following results:

![Thresholded Digits](https://i.imgur.com/mPjBVmH.png)

## display_hamming_distance
Calculates the hamming distances between the thresholded digits and displays them using matplotlib as shown in the image below.

![Hamming Distances](https://i.imgur.com/V8NJKvW.png)

## get_closest_digits
This prints a a list of the closest digits in descending order based on hamming distance for each digit.

To execute any of the methods either import them into a script and call them or call them at the end of mnist_util.py and execute the file. An example is given below:

Add display_hamming_distance to the end of `mnist_util.py`
```
mnist_util.py
.
.
.
 display_hamming_distance()
```
Execute `python mnist_util.py` in your terminal.

---
---
# Usage: `draw_and_classify.py`

To run this application simply execute the following in your terminal:
```
python draw_and_classify.py
```

This should display a window as shown below:

![Hamming Distances](https://i.imgur.com/AJqsEb2.png)

You can then use your mouse to draw any digit you like as shown in the example below:

![Hamming Distances](https://i.imgur.com/LDWB2sZ.png)
### Controls

```
You can use the 'c' key to clear the drawing.
```
```
You can use the 'g' key to copy the drawing in the format required to pass it as an argument to the `classify` method in main.py
```
```
You can use the 'h' key to directly classify the drawing using the hopfield network implementations
```

If you click 'h' the result of the classification should be display using matplotlib as shown below:

![Hamming Distances](https://i.imgur.com/vb1tbIo.png)


