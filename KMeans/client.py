import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from kmeans import KMeans


# get the synthetic dataset
X, Y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1);

Y = np.ones(shape=Y.shape) - Y; # done only for visualization purposes
num_clusters = 2;
kmean = KMeans(num_clusters=num_clusters);
cluster_centers, cluster_assignments = kmean.fit(X, num_epochs=10);

# plot the data and the clustered data
plt.figure(figsize=(8,8));
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9);
plt.subplot(311)
plt.title('Plot for the unlabeled data', fontsize='small');
plt.scatter(X[:,0], X[:,1], s=25, c=None);

plt.subplot(312);
plt.title('Plot for the clustered data', fontsize='small');
plt.scatter(X[:,0], X[:,1], s=25, c=cluster_assignments * 2);

plt.subplot(313);
plt.title('Plot for the Ground truth labeling data', fontsize='small');
plt.scatter(X[:,0], X[:,1], s=25, c=Y);

plt.show();
