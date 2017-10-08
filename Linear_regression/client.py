# load the boston housing prices dataset
from sklearn.datasets import load_boston
from linear_regression import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from sklearn import preprocessing


# load the dataset
X_train, Y_train = load_boston(return_X_y=True);

# create the linear regression model object
model = LinearRegression();

# create the list of the regularisation parameters to be used
reg_param = [0.1, 1.0, 10.0, 100.0, 1000.0];
num_epochs = 100000;

model_params_ls = model.fit('ls', X_train, Y_train);
print(model_params_ls);

# compute the lasso parameters for different reg param values
# and plot the values in a graph
plt.figure(1);
plt.title('LASSO parameter profiles at different regularisation params');
plt.xlabel('parameter index');
plt.ylabel('parameter values');
for i in range(len(reg_param)):

    model_params_lasso = model.fit('lasso', X_train, Y_train, 
                                    regularization_param=reg_param[i], 
                                    num_epochs=num_epochs);
    plt.plot(model_params_lasso, label= '' + str(reg_param[i]));



plt.legend(loc='upper right');

# ----------------------------------------------------------------------
# compute the ridge regression parameters for different reg param values
# and plot the values in a graph
plt.figure(2);
plt.title('Ridge Regression parameter profiles at different regularisation params');
plt.xlabel('parameter index');
plt.ylabel('parameter values');
for i in range(len(reg_param)):

    model_params_rr = model.fit('rr', X_train, Y_train, regularization_param=reg_param[i]);
    plt.plot(model_params_rr, label= '' + str(reg_param[i]));

# display the plot
plt.legend(loc='upper right');
plt.show();