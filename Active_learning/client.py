import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from active_learning import ActiveLearning

# load the boston housing dataset
X, Y = load_boston(return_X_y=True);

# split the dataset to training, validation and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20);

noise_param = 1.0;
lambda_param = 1.0;

learner = ActiveLearning(X_train, X_test, Y_train, Y_test,
                         lambda_param=lambda_param,
                         noise_param=noise_param);

# start the active learning procedure
mean, covariance = learner.learn_posterior(num_epochs=100);

print(mean.shape);
print(covariance.shape);