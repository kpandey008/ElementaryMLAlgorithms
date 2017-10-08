import numpy as np
import numpy.linalg as la
import numpy.random as random
import math

class KMeans:

    def __init__(self, num_clusters=2):
 
        self.num_clusters = num_clusters;
    
    def fit(self, X, num_epochs=100):

        # fits the K-means model on the data to find the clusters

        num_samples = X.shape[0];
        data_dimension = X.shape[1];
        # initialize the cluster centers randomly
        cluster_centers = random.randn(self.num_clusters, data_dimension);
        cluster_assigments = np.zeros(shape=(num_samples,));

        for epoch in range(num_epochs):

            # the logic for the fit goes here
            # parameter update logic: Coordinate Gradient Descent


            """ Step 1: Update the cluster assignments """
            for sample_idx in range(num_samples):

                current_sample = X[sample_idx];

                minDist = self.computeDistance(cluster_centers[0,:], current_sample);
                cluster_assignment = 0;
                for cluster_idx in range(self.num_clusters):

                    current_cluster = cluster_centers[cluster_idx,:];
                    dist = self.computeDistance(current_cluster, current_sample);

                    if(dist < minDist):
                        cluster_assignment = cluster_idx;
                
                # update the cluster assignment
                cluster_assigments[sample_idx] = cluster_assignment;

            """ Step 2: Update the cluster centers using the above assignments"""
            for cluster_idx in range(self.num_clusters):

                sum_cluster_elements = np.zeros(shape=(data_dimension,));
                num_cluster_elements = 0;
                for sample_idx in range(num_samples):
                    
                    if(cluster_assigments[sample_idx] == cluster_idx):
                        sum_cluster_elements = sum_cluster_elements + X[sample_idx];
                        num_cluster_elements += 1;
                
                cluster_centers[cluster_idx] = sum_cluster_elements/num_cluster_elements;
            
            """Calculate the value for the error"""
            error = 0.0
            for sample_idx in range(num_samples):
                for cluster_idx in range(self.num_clusters):
                    if(cluster_assigments[sample_idx] == cluster_idx):
                        error += la.norm(X[sample_idx] - cluster_centers[cluster_idx], ord=2);

            print('Epoch:%d Error:%0.2f' % (epoch, error));
            
        return (cluster_centers, cluster_assigments);
        
    
    def computeDistance(self,x1, x2):

        distance = la.norm(x1 - x2, ord=2);
        return distance;