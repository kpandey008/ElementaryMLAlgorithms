# the code is the demonstration of the linear regression
# TODO - Implement LARS and proximal gradient descent methods for LASSO

# import the required python libraries
import numpy as np
import numpy.linalg as la
from sklearn import preprocessing

class LinearRegression:

    def fit(self, key, X_train, Y_train, regularization_param=0, num_epochs=10):

        # fit the algorithm provided the training data
        params = [];
        if(key == 'ls'):
            params = self.fit_ls(X_train, Y_train);
        elif(key == 'rr'):
            params = self.fit_rr(X_train, Y_train, regularization_param);
        elif(key == 'lasso'):
            params = self.fit_lasso(X_train, Y_train, regularization_param, num_epochs);
        else:
            raise ValueError('Please provide the correct value of the ley argument');

        return params;

    
    def fit_ls(self, X_train, Y_train):

        print('------------Using Linear regression with Least squares model-------------------');
        print('Model training data shape:', X_train.shape);
        print('Model training data shape:', Y_train.shape);

        # determine the paramters of the model using linear regression least sqaures
        dd_correlation = np.matmul(np.transpose(X_train), X_train);
        dl_correlation = np.matmul(np.transpose(X_train), Y_train);
        params = np.matmul(la.inv(dd_correlation), dl_correlation);

        return params;

    def fit_rr(self, X_train, Y_train, lambda_param):

        print('----------Using Linear regression with L2 regularization----------------------');
        print('Model training data shape:', X_train.shape);
        print('Model training data shape:', Y_train.shape);

        # determine the paramters of the model using linear regression with L2 regularization
        # note that a close form solution exists for this type of regression.

        tmp = np.matmul(np.transpose(X_train), X_train);
        dd_correlation = np.add(tmp, lambda_param* np.identity(tmp.shape[0]));

        dl_correlation = np.matmul(np.transpose(X_train), Y_train);
        params = np.matmul(la.inv(dd_correlation), dl_correlation);

        return params;


    def fit_lasso(self, X_train, Y_train, lambda_param, num_epochs):

        print('----------Using Linear regression with L1 regularization----------------------');
        print('Model training data shape:', X_train.shape);
        print('Model training data shape:', Y_train.shape);

        learning_rate = 0.00001;
        print('Learning rate:', learning_rate);

        # determine the paramters of the model using linear regression with L1 regularization
        # note that a close form solution does not exists for this type of regression.
        # use the vanilla gradient descent method. Using the vanilla gradient descent gives only
        # approximate sparse solutions(not completely zero)

        X_train = preprocessing.scale(X_train);
        Y_train = preprocessing.scale(Y_train);

        params = np.ones(X_train.shape[1]);

        num_samples = X_train.shape[1];

        for i in range(num_epochs):

            # compuet the loss on the prediction
            rss_loss = Y_train - np.matmul(X_train, params);
            mse = la.norm(rss_loss, ord=2) + lambda_param * la.norm(params, ord=1);
            # print mse;
            # compute the gradient
            # notice the subgradient used in the computation of the gradient of L1 norm
            gradient = (-np.dot(np.transpose(X_train), rss_loss) + \
                        lambda_param * np.sign(params))/num_samples;

            # update the parameters
            params = params - learning_rate * gradient;

        # print params;
        return params;