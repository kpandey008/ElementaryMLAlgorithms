import numpy as np
import numpy.linalg as la
import math

class ActiveLearning:

    def __init__(self, X_train, X_test, Y_train, Y_test, lambda_param=1.0, noise_param=1.0):

        self.X_train = X_train;
        self.X_test = X_test;

        self.Y_train = Y_train;
        self.Y_test = Y_test;

        self.lambda_param = lambda_param;
        self.noise_param = noise_param;

    def compute_posterior_distribution(self, X, Y):

        # used to compute the posterior distribution for the MAP inference
        # This represents the interpretation of RR from a Bayesian perspective

        data_correlation = np.matmul(np.transpose(X), X);
        data_label_correlation = np.matmul(np.transpose(X), Y);

        tmp = self.lambda_param * self.noise_param * np.identity(data_correlation.shape[0]);
        tmp2 = self.lambda_param * np.identity(data_correlation.shape[0]);
        
        # compute the mean and the covariance of the data
        mean = np.matmul(la.inv(tmp + data_correlation), data_label_correlation);
        covariance = la.inv(tmp2 + data_correlation/self.noise_param);

        return (mean, covariance);

    def compute_predictive_distribution(self, x, mean, covariance):

        # used to compute the predictive distribution for the Regression framework
        # This represents the prediction on a new data point with a measure of uncertainity in the prediction

        # x is a single data point as represented by a lower case representation

        prediction = np.dot(x, mean);

        tmp = np.matmul(covariance, x);
        prediction_uncertainity = self.noise_param + np.dot(x,tmp);

        return (prediction, prediction_uncertainity);

    
    def learn_posterior(self, num_epochs=10):

        # employs the active learning iterative technique to learn the posterior efficiently
        # This reduces the entropy of the posterior distribution that we learn over the dataset
        # This type of learning can also be used for semi supervised learning problem

        mean , covariance = 0, 0;

        for i in range(num_epochs):

            # estimate the posterior distribution over the entire training data
            mean , covariance = self.compute_posterior_distribution(self.X_train, self.Y_train);

            # compute the entropy of the posterior
            entropy = np.log(2 * math.pi * math.e * la.det(covariance));
            print(entropy);

            max_sigma = 0.0;
            max_index = -1;
            max_prediction = 0.0;

            # estimate the predictive distribution over the entire testing dataset
            for i in range(len(self.X_test)):
                current_sample = self.X_test[i];
                prediction , prediction_uncertainity = self.compute_predictive_distribution(current_sample,
                                                                                    mean,
                                                                                    covariance);
                if(prediction_uncertainity > max_sigma):
                    max_index = i;
                    max_sigma = prediction_uncertainity;
                    max_prediction = prediction;
            
            # append the data point to the training set
            self.X_train = np.append(self.X_train, 
                                    np.reshape(self.X_test[max_index],newshape=(-1,self.X_test[max_index].shape[0])),
                                    axis=0);
            self.Y_train = np.append(self.Y_train, max_prediction);
        
        return (mean, covariance);
