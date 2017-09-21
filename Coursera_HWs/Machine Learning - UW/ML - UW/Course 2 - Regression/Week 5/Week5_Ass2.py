"""
Week 5 - Assignment 2
"""
import graphlab
import numpy as np
import math
from math import log, sqrt
import pprint

sales = graphlab.SFrame('kc_house_data.gl/')
# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int) 
"""
If we want to do any "feature engineering" 
like creating new features or adjusting 
existing ones we should do this directly 
using the SFrames as seen in the first 
notebook of Week 2. For this notebook, 
however, we will work with the existing 
features.
"""

"""
\* Import useful functions from previous notebook *\

As in Week 2, we convert the 
SFrame into a 2D Numpy array. 
Copy and paste get_numpy_data() 
from the second notebook of Week 2.
"""
def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of 
    # the features list so that we can extract 
    # it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame 
    # given by the features list into 
    # the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]

    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe 
    # associated with the output to the SArray output_sarray
    output_sarray = data_sframe[output]
    # the following will convert the SArray 
    # into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)


def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)

"""
\*   Normalize features    */

In the house dataset, features vary 
wildly in their relative magnitude: 
sqft_living is very large overall 
compared to bedrooms, for instance. 

As a result, weight for sqft_living 
would be much smaller than weight 
for bedrooms. This is problematic 
because "small" weights are dropped 
first as l1_penalty goes up.

To give equal considerations for all 
features, we need to normalize features 
as discussed in the lectures: we divide 
each feature by its 2-norm so that 
the transformed feature has norm 1.

Let's see how we can do this 
normalization easily with Numpy: 
let us first consider a small matrix.
"""
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    return (feature_matrix / norms, norms)

"""
# Test above function
features, norms = normalize_features(np.array([[3.,6.,9.],[4.,8.,12.]]))
print features
# should print
# [[ 0.6  0.6  0.6]
#  [ 0.8  0.8  0.8]]
print norms
# should print
# [5.  10.  15.]
"""

"""
Implementing Coordinate Descent with normalized features

We seek to obtain a sparse set of weights by minimizing the LASSO cost function

SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|).

(By convention, we do not include 
w[0] in the L1 penalty term. 
We never want to push the intercept to zero.)

The absolute value sign makes 
the cost function non-differentiable, 
so simple gradient descent is not 
viable (you would need to implement 
a method called subgradient descent). 
Instead, we will use coordinate descent: 
at each iteration, we will fix all weights 
but weight i and find the value of 
weight i that minimizes the objective. 
That is, we look for 
argmin_{w[i]} [ SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|) ]

where all weights other than w[i] are held to be constant.

We will optimize one w[i] at a time, 
circling through the weights multiple times.

1- Pick a coordinate i
2- Compute w[i] that minimizes the cost 
   function SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|)
3- Repeat Steps 1 and 2 for all coordinates, multiple times.


For this notebook, we use cyclical 
coordinate descent with normalized 
features, where we cycle through 
coordinates 0 to (d-1) in order, 
and assume the features were 
normalized as discussed above. 
The formula for optimizing each coordinate is as follows:
#        | (ro[i] + lambda/2)     if ro[i] < -lambda/2
# w[i] = | 0                      if -lambda/2 <= ro[i] <= lambda/2
#        | (ro[i] - lambda/2)     if ro[i] > lambda/2

where
ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ].
Note that we do not regularize 
the weight of the constant 
feature (intercept) w[0], so, 
for this weight, the update is simply:
w[0] = ro[i]
"""

"""
\*   Effect of L1 penalty    */

Let us consider a simple model with 2 features:
"""
simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
simple_feature_matrix, norms = normalize_features(simple_feature_matrix)
# We assign some random set of initial weights and inspect the values of ro[i]:
weights = np.array([1., 4., 1.])

prediction = predict_output(simple_feature_matrix, weights)

"""
Compute the values of ro[i] 
for each feature in this simple model, 
using the formula given above, 
using the formula:
ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]
Hint: You can get a Numpy vector 
for feature_i using:
simple_feature_matrix[:,i]
"""

(no_data, no_features) = simple_feature_matrix.shape
ro_vector = np.zeros((no_features, 1))
error = output - prediction
for feat_count in xrange(0, no_features):
    ro_vector[feat_count] = np.sum(simple_feature_matrix[:, feat_count] * 
                                  (error + weights[feat_count] * simple_feature_matrix[:,feat_count]))
# print "ro_vector =", ro_vector

"""
\*  Single Coordinate Descent Step  */

Using the formula above, 
implement coordinate descent 
that minimizes the cost function 
over a single feature i. Note 
that the intercept (weight 0) 
is not regularized. The function 
should accept feature matrix, 
output, current weights, 
l1 penalty, and index of 
feature to optimize over. 

The function should return 
new weight for feature i.
"""    
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    predictions = predict_output(feature_matrix, weights)
    error = output - predictions
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = np.sum(feature_matrix[:, i] * (error + weights[i] * feature_matrix[:,i]))
    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i 
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2.
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2.
    else:
        new_weight_i = 0.

    return new_weight_i
"""
# test: should print 0.425558846691
print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]), 
                                   np.array([1., 1.]), np.array([1., 4.]), 0.1)
"""

"""
\*  Cyclical coordinate descent  */

Now that we have a function that 
optimizes the cost function over 
a single coordinate, let us 
implement cyclical coordinate 
descent where we optimize 
coordinates 0, 1, ..., (d-1) in 
order and repeat.

When do we know to stop? Each 
time we scan all the coordinates
(features) once, we measure the 
change in weight for each 
coordinate. If no coordinate changes 
by more than a specified threshold, 
we stop.

For each iteration:
   1- As you loop over features 
      in order and perform coordinate 
      descent, measure how much each 
      coordinate changes.

   2- After the loop, if the maximum 
      change across all coordinates 
      is falls below the tolerance, 
      stop. Otherwise, go back to step 1.
Return weights

IMPORTANT: when computing a new 
           weight for coordinate i, 
           make sure to incorporate 
           the new weights for coordinates 
           0, 1, ..., i-1. One good way 
           is to update your weights
        variable in-place. See following 
        pseudocode for illustration.
     
        for i in range(len(weights)):
        old_weights_i = weights[i] # remember 
        old value of weight[i], as it 
        will be overwritten
          # the following line uses 
            new values for weight[0], weight[1], ..., weight[i-1]
          # and old values for weight[i], ..., weight[d-1]
          weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
          # use old_weights_i to compute change in coordinate
"""
def lasso_cyclical_coordinate_descent(feature_matrix, 
                                      output, 
                                      initial_weights, 
                                      l1_penalty, 
                                      tolerance):
    (no_data, no_features) = feature_matrix.shape
    weights = np.array(initial_weights)
    weight_changes = np.array(initial_weights) * 0.0
    converged = False
    while not converged:
        for i in xrange(0, no_features):
            new_weight_i = lasso_coordinate_descent_step(i, feature_matrix, 
                                                         output, weights, l1_penalty)
            weight_changes[i] = np.abs(new_weight_i - weights[i])
            weights[i] = new_weight_i
        max_change = np.max(weight_changes)
        if max_change < tolerance:
            converged = True
    return weights
# Using the following parameters, learn the weights on the sales dataset.
simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0    
# First create a normalized version of the feature matrix, normalized_simple_feature_matrix.    
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
 # normalize features:
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix)

# Then, run your implementation of LASSO coordinate descent:
weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)

# print "weights =", weights
predictions = predict_output(normalized_simple_feature_matrix, weights)
RSS = np.sum((output - predictions) ** 2)
# print "RSS =", RSS

"""
\*  Evaluating LASSO fit with more features  */
Let us split the sales dataset into training and test sets.
"""
train_data,test_data = sales.random_split(.8,seed=0)
all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']
output = 'price'

(feature_matrix, output) = get_numpy_data(train_data, all_features, output)
(normal_train_feat_matrix, train_norms) = normalize_features(feature_matrix)
"""
First, learn the weights with 
l1_penalty=1e7, on the training data. 
Initialize weights to all zeros, and 
set the tolerance=1. 
Call resulting weights weights1e7, you will need them later.
"""
l1_penalty = 1e7
initial_weights = np.zeros((feature_matrix.shape[1]))
tolerance = 1
weights1e7 = lasso_cyclical_coordinate_descent(normal_train_feat_matrix, output,
                                               initial_weights, l1_penalty, tolerance)
# print "weights1e7 =", weights1e7 
"""
Learn this:

feature_list = ['constant'] + all_features
print feature_list
feature_weights1e7 = dict(zip(feature_list, weights1e7))
for k,v in feature_weights1e7.iteritems():
    if v != 0.0:
        print k, v
"""

l1_penalty = 1e8
initial_weights = np.zeros((feature_matrix.shape[1]))
tolerance = 1
weights1e7 = lasso_cyclical_coordinate_descent(normal_train_feat_matrix, output,
                                               initial_weights, l1_penalty, tolerance)

l1_penalty = 1e4
tolerance = 5e5
initial_weights = np.zeros((feature_matrix.shape[1]))
weights1e4 = lasso_cyclical_coordinate_descent(normal_train_feat_matrix, output,
                                               initial_weights, l1_penalty, tolerance)

(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')
prediction =  predict_output(test_feature_matrix, normalized_weights1e7)
RSS = np.dot(test_output-prediction, test_output-prediction)
print 'RSS for model with weights1e7 = ', RSS


prediction =  predict_output(test_feature_matrix, normalized_weights1e8)
RSS = np.dot(test_output-prediction, test_output-prediction)
print 'RSS for model with weights1e8 = ', RSS


prediction =  predict_output(test_feature_matrix, normalized_weights1e4)
RSS = np.dot(test_output-prediction, test_output-prediction)
print 'RSS for model with weights1e4 = ', RSS