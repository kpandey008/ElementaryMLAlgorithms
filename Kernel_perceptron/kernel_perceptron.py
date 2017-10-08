import numpy as np
from numpy import linalg as la
import math

class Kernel_perceptron:

    def __init__(self):

        # to store all the misclassified training samples
        self.falsely_classified = []; 
        self.kernel = 'rbf';
    
    def fit(self, X_train, Y_train, num_classes, kernel='rbf'):

        # used to fit the perceptron with the training samples
        
        self.kernel = kernel;

        if(len(self.falsely_classified) == 0):
            self.falsely_classified.append((X_train[0], Y_train[0]));
            
        # find all the misclassified training samples
        num_samples = X_train.shape[0];
        num_epochs = 4;
        correctly_predicted = 0;

        for epoch in range(num_epochs):

            correctly_predicted = 0;
            for i in range(num_samples):

                similiarity = np.zeros(shape=(len(self.falsely_classified),));
                max_similiarity = 0;
                predicted_class = 0;

                for j in range(len(self.falsely_classified)):

                    similiarity_index = self.calculate_kernel_val(self.kernel, X_train[i], self.falsely_classified[j][0]);
                    similiarity[j] = similiarity_index;

                    if(similiarity_index > max_similiarity):
                        max_similiarity = similiarity_index;
                        predicted_class = self.falsely_classified[j][1];
                    

                if(predicted_class != Y_train[i]):
                    # add the sample to the list of misclassified samples
                    self.falsely_classified.append((X_train[i],Y_train[i]));
                else:
                    correctly_predicted += 1;
    
    def predict(self, X_test, Y_test):

        # returns the accuracy of the classifier
        correctly_predicted = 0.0;
        num_test_samples = X_test.shape[0];

        for i in range(num_test_samples):

            similiarity = np.zeros(shape=(len(self.falsely_classified),));
            max_similiarity = 0;
            predicted_class = 0;

            for j in range(len(self.falsely_classified)):

                similiarity_index = self.calculate_kernel_val(self.kernel, X_test[i], self.falsely_classified[j][0]);
                similiarity[j] = similiarity_index;

                if(similiarity_index > max_similiarity):
                    max_similiarity = similiarity_index;
                    predicted_class = self.falsely_classified[j][1];
                    

            if(predicted_class == Y_test[i]):
                correctly_predicted += 1;
        
        accuracy = correctly_predicted/num_test_samples;
        return accuracy;


    def calculate_kernel_val(self, kernel, sample1, sample2):

        if(kernel == 'rbf'): 
            kernel_radius = 0.2; #This is a kernel hyperparameter
            return np.exp(-la.norm(sample1 - sample2, ord=2)/(math.pow(kernel_radius,2))); 
        else:
            raise ValueError('Not been implemented yet');