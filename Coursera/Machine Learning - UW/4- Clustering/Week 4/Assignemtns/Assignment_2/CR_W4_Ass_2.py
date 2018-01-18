# Clustering and Retrieval - Week 4 - Assignment 2
import graphlab
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances
import time
import sys
import os
import copy
import array

from sklearn.cluster import KMeans

"""
We also have a Python file containing 
implementations for several functions 
that will be used during the course of 
this assignment.
"""
from em_utilities import *

"""
## Fitting a diagonal covariance Gaussian mixture model to text data

In a previous assignment, we explored k-means clustering for a high-dimensional Wikipedia dataset. We can also model this data with a mixture of Gaussians, though with increasing dimension we run into two important issues associated with using a full covariance matrix for each component.
 * Computational cost becomes prohibitive in high dimensions: score calculations have complexity cubic in the number of dimensions M if the Gaussian has a full covariance matrix.
 * A model with many parameters require more data: observe that a full covariance matrix for an M-dimensional Gaussian will have M(M+1)/2 parameters to fit. With the number of parameters growing roughly as the square of the dimension, it may quickly become impossible to find a sufficient amount of data to make good inferences.

Both of these issues are avoided if we require the covariance matrix of each component to be diagonal, as then it has only M parameters to fit and the score computation decomposes into M univariate score calculations. Recall from the lecture that the M-step for the full covariance is:

\begin{align*}
\hat{\Sigma}_k &= \frac{1}{N_k^{soft}} \sum_{i=1}^N r_{ik} (x_i-\hat{\mu}_k)(x_i - \hat{\mu}_k)^T
\end{align*}

Note that this is a square matrix with M rows and M columns, and the above equation implies that the (v, w) element is computed by

\begin{align*}
\hat{\Sigma}_{k, v, w} &= \frac{1}{N_k^{soft}} \sum_{i=1}^N r_{ik} (x_{iv}-\hat{\mu}_{kv})(x_{iw} - \hat{\mu}_{kw})
\end{align*}

When we assume that this is a diagonal matrix, then non-diagonal elements are assumed to be zero and we only need to compute each of the M elements along the diagonal independently using the following equation. 

\begin{align*}
\hat{\sigma}^2_{k, v} &= \hat{\Sigma}_{k, v, v}  \\
&= \frac{1}{N_k^{soft}} \sum_{i=1}^N r_{ik} (x_{iv}-\hat{\mu}_{kv})^2
\end{align*}

In this section, we will use an EM implementation to fit a Gaussian mixture model with **diagonal** covariances to a subset of the Wikipedia dataset. The implementation uses the above equation to compute each variance term. 

We'll begin by importing the dataset and coming up with a useful representation for each article. After running our algorithm on the data, we will explore the output to see whether we can give a meaningful interpretation to the fitted parameters in our model.
"""

wiki = graphlab.SFrame('people_wiki.gl/').head(5000)
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['text'])

tf_idf, map_index_to_word = sframe_to_scipy(wiki, 'tf_idf')
tf_idf = normalize(tf_idf)

"""
for i in range(5):
    doc = tf_idf[i]
    print(np.linalg.norm(doc.todense()))
"""

## EM in high dimensions
"""
EM for high-dimensional data requires some special treatment:
 * E step and M step must be vectorized as much as possible, as explicit loops are dreadfully slow in Python.
 * All operations must be cast in terms of sparse matrix operations, to take advantage of computational savings enabled by sparsity of data.
 * Initially, some words may be entirely absent from a cluster, causing the M step to produce zero mean and variance for those words.  This means any data point with one of those words will have 0 probability of being assigned to that cluster since the cluster allows for no variability (0 variance) around that count being 0 (0 mean). Since there is a small chance for those words to later appear in the cluster, we instead assign a small positive variance (~1e-10). Doing so also prevents numerical overflow.
 
We provide the complete implementation for you in the file `em_utilities.py`. For those who are interested, you can read through the code to see how the sparse matrix implementation differs from the previous assignment. 

You are expected to answer some quiz questions using the results of clustering.
"""

### Initializing mean parameters using k-means
"""
Recall from the lectures that EM for Gaussian mixtures is very sensitive to the choice of 
initial means. With a bad initial set of means, EM may produce clusters that span a large 
area and are mostly overlapping. To eliminate such bad outcomes, we first produce a 
suitable set of initial means by using the cluster centers from running k-means.  That is,
we first run k-means and then take the final set of means from the converged solution as 
the initial means in our EM algorithm.
"""


np.random.seed(5)
num_clusters = 25

# Use scikit-learn's k-means to simplify workflow
#kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1) # uncomment to use parallelism -- may break on your installation
kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=1)
kmeans_model.fit(tf_idf)
centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_

means = [centroid for centroid in centroids]


### Initializing cluster weights**
"""
We will initialize each cluster 
weight to be the proportion of 
documents assigned to that 
cluster by k-means above.
"""
num_docs = tf_idf.shape[0]
weights = []
for i in xrange(num_clusters):
    # Compute the number of data points assigned to cluster i:
    num_assigned = cluster_assignment[cluster_assignment == i].shape[0] # YOUR CODE HERE
    w = float(num_assigned) / num_docs
    weights.append(w)

### Initializing covariances**
"""
To initialize our covariance parameters, we compute 
$\hat{\sigma}_{k, j}^2 = \sum_{i=1}^{N}(x_{i,j} - \hat{\mu}_{k, j})^2$ 
for each feature $j$.
For features with really tiny variances, 
we assign 1e-8 instead to prevent numerical 
instability. We do this computation in a 
vectorized fashion in the following code block.
"""
covs = []
for i in xrange(num_clusters):
    member_rows = tf_idf[cluster_assignment==i]
    cov = (member_rows.multiply(member_rows) - 2*member_rows.dot(diag(means[i]))).sum(axis=0).A1 / member_rows.shape[0] \
          + means[i]**2
    cov[cov < 1e-8] = 1e-8
    covs.append(cov)


### Running EM
# Now that we have initialized all of our parameters, run EM.

out = EM_for_high_dimension(tf_idf, means, covs, weights, cov_smoothing=1e-10)
print "out['loglik']"
print out['loglik']

## Interpret clustering results
"""
In contrast to k-means, EM is able to explicitly model clusters of varying sizes and proportions. The relative magnitude of variances in the word dimensions tell us much about the nature of the clusters.

Write yourself a cluster visualizer as follows.  Examining each cluster's mean vector, list the 5 words with the largest mean values (5 most common words in the cluster). For each word, also include the associated variance parameter (diagonal element of the covariance matrix). 

A sample output may be:
```
==========================================================
Cluster 0: Largest mean parameters in cluster 

Word        Mean        Variance    
football    1.08e-01    8.64e-03
season      5.80e-02    2.93e-03
club        4.48e-02    1.99e-03
league      3.94e-02    1.08e-03
played      3.83e-02    8.45e-04
...
```
"""

# Fill in the blanks
def visualize_EM_clusters(tf_idf, means, covs, map_index_to_word):
    print('')
    print('==========================================================')

    num_clusters = len(means)
    for c in xrange(num_clusters):
        print('Cluster {0:d}: Largest mean parameters in cluster '.format(c))
        print('\n{0: <12}{1: <12}{2: <12}'.format('Word', 'Mean', 'Variance'))
        
        # The k'th element of sorted_word_ids should be the index of the word 
        # that has the k'th-largest value in the cluster mean. Hint: Use np.argsort().
        sorted_word_ids = np.argsort(-means[c])

        for i in sorted_word_ids[:5]:
            print '{0: <12}{1:<10.2e}{2:10.2e}'.format(map_index_to_word['category'][i], 
                                                       means[c][i],
                                                       covs[c][i])
        print '\n=========================================================='


'''By EM'''
# visualize_EM_clusters(tf_idf, out['means'], out['covs'], map_index_to_word)

## Comparing to random initialization
"""
Create variables for randomly initializing the EM algorithm. Complete the following code block.
"""
np.random.seed(5) # See the note below to see why we set seed=5.
num_clusters = len(means)
num_docs, num_words = tf_idf.shape

random_means = []
random_covs = []
random_weights = []

for k in range(num_clusters):
    
    # Create a numpy array of length num_words with random normally distributed values.
    # Use the standard univariate normal distribution (mean 0, variance 1).
    # YOUR CODE HERE
    mean = np.random.normal(0, 1, num_words)
    
    # Create a numpy array of length num_words with random values uniformly distributed between 1 and 5.
    # YOUR CODE HERE
    cov = np.random.uniform(1,6,num_words)

    # Initially give each cluster equal weight.
    # YOUR CODE HERE
    weight = 1
    
    random_means.append(mean)
    random_covs.append(cov)
    random_weights.append(weight)

out_random_init = EM_for_high_dimension(tf_idf, random_means, random_covs, random_weights, cov_smoothing=1e-5)

print "out_random_init['loglik']"
print out_random_init['loglik']

# YOUR CODE HERE. Use visualize_EM_clusters, which will require you to pass in tf_idf and map_index_to_word.
visualize_EM_clusters(tf_idf, out_random_init['means'], out_random_init['covs'], map_index_to_word)

"""
**Note**: Random initialization may sometimes produce a superior fit than k-means 
initialization. We do not claim that random initialization is always worse. However, 
this section does illustrate that random initialization often produces much worse clustering 
than k-means counterpart. This is the reason why we provide the particular random seed 
(`np.random.seed(5)`).
"""


## Takeaway
"""
In this assignment we were 
able to apply the EM algorithm 
to a mixture of Gaussians model 
of text data. This was made 
possible by modifying the model 
to assume a diagonal covariance
for each cluster, and by modifying 
the implementation to use a sparse 
matrix representation. 
In the second part you explored the 
role of k-means initialization on 
the convergence of the model as well 
as the interpretability of the clusters.
"""




