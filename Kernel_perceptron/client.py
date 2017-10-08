# implemented only the kernel perceptron for now..can also implement K-nn, and LWR
# Can also use many other kernel functions.

import numpy as np
import numpy.linalg as la
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from kernel_perceptron import Kernel_perceptron


# get the iris dataset for classification
X , Y = load_iris(return_X_y=True);

# separate the train and the test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4);

perceptron = Kernel_perceptron();
perceptron.fit(X_train, Y_train, 3, kernel='rbf');

accuracy = perceptron.predict(X_test, Y_test);
print('Accuracy:', accuracy);