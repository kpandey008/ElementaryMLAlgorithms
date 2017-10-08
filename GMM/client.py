import numpy as np
from sklearn.datasets import make_classification
from sklearn import preprocessing
import matplotlib.pyplot as plt
from GMM import GMM
from sklearn import mixture


# generate the dataset
X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=2);

X = preprocessing.scale(X);

num_clusters = 3;
num_epochs = 50;

gmm_model = GMM();
phi, pi_dist, mean, covariance = gmm_model.fit(X, num_clusters=num_clusters, num_epochs=num_epochs);

gmm_sklearn = mixture.GaussianMixture(n_components=2);
gmm_sklearn.fit(X);
plt.figure(figsize=(8,8));
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9);

plt.subplot(211);
plt.title('Plot for the unclustered data', fontsize='small');
plt.scatter(X[:,0],X[:,1],s=25, c=None);

plt.subplot(212);
plt.title('Plot for the clustered data', fontsize='small');
plt.scatter(X[:,0], X[:,1], s=25, c=phi);

plt.show();
