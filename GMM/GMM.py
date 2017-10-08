import numpy as np
import numpy.linalg as la
import numpy.random as random
import math

class GMM:

    def fit(self, X, num_clusters=2, num_epochs=50):

        # describes the training criteria for the GMM using EM formulation
        num_samples = X.shape[0];
        num_dims = X.shape[1];

        """Step1 : Initialize the gaussian parameters and the discrete distribution"""
        pi_distribution = random.rand(num_clusters,);
        pi_distribution = pi_distribution / la.norm(pi_distribution, ord=1);

        print pi_distribution;
    
        mean = random.randn(num_clusters, num_dims);
        covariance = random.randn(num_clusters, num_dims, num_dims);

        phi = np.zeros(shape=(num_samples, num_clusters));

        prev_error = 0.0;
        new_error = 0.0;

        epsilon = 0.0001;

        for epoch_idx in range(num_epochs):

            """Step2: Assign the data to the clusters"""
            for sample_idx in range(num_samples):

                for cluster_idx in range(num_clusters):
                    phi[sample_idx, cluster_idx] = pi_distribution[cluster_idx] * \
                                                    self.compute_gaussian(np.reshape(X[sample_idx],(1,2)), 
                                                                          np.reshape(mean[cluster_idx],(1,2)),
                                                                          covariance[cluster_idx]);
            
                # normalize the distribution over the weights
                phi[sample_idx] = phi[sample_idx] / la.norm(phi[sample_idx], ord=1);
            
            """Step3: Update the parameters of the model """
            # update the pi distribution
            nk = np.sum(phi, axis=0);
            pi_distribution = nk / num_samples;

            # update the mean of the clusters
            mean = np.matmul(np.transpose(phi), X);
            for cluster_idx in range(num_clusters):
                mean[cluster_idx] = mean[cluster_idx] / nk[cluster_idx];

            # update the covariance of the clusters
            for cluster_idx in range(num_clusters):

                sample = np.zeros(shape=(num_dims, num_dims));
                for sample_idx in range(num_samples):

                    tmp = X[sample_idx] - mean[cluster_idx];
                    tmp = np.reshape(tmp, (1,2));

                    sample = sample + phi[sample_idx,cluster_idx] * np.matmul(np.transpose(tmp), tmp);

                covariance[cluster_idx] = sample / nk[cluster_idx];
            
            # compute the error at the end of each iteration
            error = 0;
            for sample_idx in range(num_samples):
                
                sample_error = 0.0;
                for cluster_idx in range(num_clusters):
                    sample_error += (pi_distribution[cluster_idx] * 
                              self.compute_gaussian(np.reshape(X[sample_idx],(1,2)),
                                                    np.reshape(mean[cluster_idx],(1,2)),
                                                    covariance[cluster_idx]));

                error += math.log(sample_error);
            
            prev_error = new_error;
            new_error = error;

            print ('Log likelihood:%0.5f at Epoch:%d' % (error, epoch_idx));                                    
            print pi_distribution;
            if abs(new_error - prev_error) < epsilon:
                break;

        # return the theta,mean and the covariance
        return (phi, pi_distribution, mean, covariance);

    def compute_gaussian(self, x, mean, covariance):

        # computes the rbf distance of x using the mean and the covariance
        cov_inv = la.inv(covariance);
        tmp = np.matmul(x - mean , cov_inv);
        dist = np.exp(-np.matmul(tmp, np.transpose(x - mean)))/(math.sqrt(2 * np.pi) * np.abs(la.det(covariance)));

        return dist;