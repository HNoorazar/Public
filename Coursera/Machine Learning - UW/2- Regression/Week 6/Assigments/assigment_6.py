"""
Regression. UW. Week 6.
\*   Predicting house prices using k-nearest neighbors regression   */

In this notebook, you will 
implement k-nearest neighbors 
regression. You will:
Find the k-nearest neighbors 
of a given query input Predict 
the output for the query input 
using the k-nearest neighbors
Choose the best value of k using 
a validation set.
"""


import graphlab
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt
import pprint


sales = graphlab.SFrame('kc_house_data_small.gl/')

"""
Import useful functions from previous notebooks
To efficiently compute pairwise 
distances among data points, 
we will convert the SFrame into 
a 2D Numpy array. First import 
the numpy library and then copy 
and paste get_numpy_data() from 
the second notebook of Week 2.
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

# Normalize features to norm 1.
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    return (feature_matrix / norms, norms)

# initial train/test split
(train_and_validation, test) = sales.random_split(.8, seed=1) 
# split training set into training and validation sets
(train, validation) = train_and_validation.random_split(.8, seed=1) 

feature_list = ['bedrooms',  
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
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')
"""
In computing distances, 
it is crucial to normalize features. 
Otherwise, for example, 
the sqft_living feature (typically on the order of thousands) 
would exert a much larger influence 
on distance than the bedrooms feature 
(typically on the order of ones). 
We divide each column of the training 
feature matrix by its 2-norm, so that 
the transformed column has unit norm.
IMPORTANT: Make sure to store the norms 
of the features in the training set. 
The features in the test and 
validation sets must be divided 
by these same norms, so that 
the training, test, and validation 
sets are normalized consistently.
"""
features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms

"""
\*   Compute a single distance   */
To start, let's just explore 
computing the "distance" between 
two given houses. We will take 
our query house to be the first 
house of the test set and look at 
the distance between this house 
and the 10th house of the training 
set.
To see the features associated 
with the query house, print the 
first row (index 0) of the test 
feature matrix. You should get 
an 18-dimensional vector whose 
components are between 0 and 1.
"""
"""
print "shape(features_train) =", np.shape(features_train)
print ""
print "features_test[0,:] =", features_test[0,:]
print ""
print "features_train[0,:] =", features_train[0,:]
print ""
print "shape(features_test[0,:]) =", np.shape(features_test[0,:])
print ""
"""
query_house = features_test[0,:]
training_10th_house = features_train[9,:]
diff = query_house - training_10th_house
distance = np.sqrt( np.sum( diff**2 ) )
print "---------------------------------"
print "question_1, distance =", distance
print "---------------------------------"
"""
\*   Compute multiple distances   */
Of course, to do nearest neighbor 
regression, we need to compute 
the distance between our query 
house and all houses in the 
training set.
To visualize this nearest-neighbor 
search, let's first compute the 
distance from our query house 
(features_test[0]) to the first 10 
houses of the training set 
(features_train[0:10]) and then 
search for the nearest neighbor 
within this small set of houses. 
Through restricting ourselves to 
a small set of houses to begin 
with, we can visually scan the 
list of 10 distances to verify 
that our code for finding the 
nearest neighbor is working.
Write a loop to compute the Euclidean 
distance from the query house to 
each of the first 10 houses in the 
training set.
"""
def find_similarities(query_house, list_of_comparisons):
    distance_vector =  np.zeros(np.shape(list_of_comparisons)[0])
    np.shape(list_of_comparisons)
    diff = query_house - list_of_comparisons
    argument = np.sum((diff**2), axis=1)
    distance_vector = np.sqrt(argument)
    return distance_vector

def find_most_similar_house(distance_vector):
    return np.min(distance_vector), np.argmin(distance_vector)

distance_vector = find_similarities(query_house, features_train[0:9, :])
print "Question 2 =", find_most_similar_house(distance_vector)
print "---------------------------------"
"""
\*   Perform 1-nearest neighbor regression   */
Now that we have the element-wise 
differences, it is not too hard 
to compute the Euclidean distances 
between our query house and all 
of the training houses. First, 
write a single-line expression 
to define a variable diff such 
that diff[i] gives the element-wise 
difference between the features of 
the query house and the i-th training house.
"""
# tests 
# diff = features_train - query_house
# print diff[-1].sum() # should print -0.0934339605842

# distance_vector = find_similarities(query_house, features_train)
# print "distance_vector[100] =", distance_vector[100] # should print 0.0237082324496


# Question 3 and 4
#"""
print "Question 3 and 4 answers:"
query_house = features_test[2]
distance_vector = find_similarities(query_house, features_train)
(distance_from_most_similar, most_similar_idx)= find_most_similar_house(distance_vector)
print "The inx of the house in the training set closest to the query house", most_similar_idx
print "predicted value for the query house is", output_train[most_similar_idx]
print "------------------------------------"
#"""

"""
\*   Perform k-nearest neighbor regression */

For k-nearest neighbors, we need to 
find a set of k houses in the training 
set closest to a given query house. 
We then make predictions based on 
these k nearest neighbors.

\*   Fetch k-nearest neighbors    */

Using the functions above, 
implement a function that takes in
   1- the value of k;
   2- the feature matrix for the training houses; and
   3- the feature vector of the query house

and returns the indices of 
the k closest  training houses. 
For instance, with 2-nearest 
neighbor, a return value of 
[5, 10] would indicate that 
the 6th and 11th training houses 
are closest to the query house.
"""
def find_k_most_similars(k, features_training, query_house):
    distance_vector = find_similarities(query_house, features_training)
    return np.argsort(distance_vector)[0:k]

# Question 5
# """
print "Questions 5"
query_house = features_test[2]
print "4 most similar houses are", find_k_most_similars(4, features_train, query_house)
print "--------------------------------------"
# """


"""
\*   Make a single prediction by averaging k nearest neighbor outputs   */

Now that we know how to find the 
k-nearest neighbors, write a 
function that predicts the value 
of a given query house. 
For simplicity, take the 
average of the prices of the k 
nearest neighbors in the training 
set. The function should have the 
following parameters:

    1- the value of k;
    2- the feature matrix for the training houses;
    3- the output values (prices) of the training houses; and
    4- the feature vector of the query house, whose price we are predicting.

The function should return a 
predicted value of the query house.

Hint: You can extract multiple 
items from a Numpy array using 
a list of indices. For instance, 
output_train[[6, 10]] returns 
the prices of the 7th and 11th 
training houses.
"""
def predict_k_most_similar_average(k, features_train, output_train, query_house):
    k_most_similar_idx = find_k_most_similars(k, features_train, query_house)
    averages = np.mean(output_train[k_most_similar_idx])
    return averages

# Question 6
#"""
query_house = features_test[2]
k = 4
prediction = predict_k_most_similar_average(k, features_train, output_train, query_house)
print "Question 6 answer"
print "The predicted price for 3rd test house is", prediction
print "--------------------------------------"
#"""

"""
\*   Make multiple predictions   */

Write a function to predict the 
value of each and every house in 
a query set. 
(The query set can be any subset 
of the dataset, be it the test set 
or validation set.) 

The idea is to have a loop where 
we take each house in the query 
set as the query house and make 
a prediction for that specific 
house. The new function should 
take the following parameters:
    the value of k;
    the feature matrix for the training houses;
    the output values (prices) of the training houses; and
    the feature matrix for the query set.

The function should return a 
set of predicted values, one 
for each house in the query set.

Hint: To get the number of houses 
in the query set, use the .shape 
field of the query features matrix.
"""

def find_predictions_for_a_set(k, features_train, output_train, query_house_list):
    predictions = np.zeros((len(query_house_list),1))
    for ii in xrange(0, len(query_house_list)):
        predictions[ii] = predict_k_most_similar_average(k, 
                                                         features_train, output_train, 
                                                         query_house_list[ii,:])
    return predictions

# Question 7

# """
query_house_list = features_test[0:10, :]
Q7 = find_predictions_for_a_set(10, features_train, output_train, query_house_list)
print "np.shape(Q7) is ", np.shape(Q7)
print "--------------------------------------"
print "Question 7 =", Q7
print "minimum index =", np.argmin(Q7)
print "minimum prediction =", np.min(Q7)
print "--------------------------------------"
# """

"""
\*    Choosing the best value of k using a validation set    */

There remains a question of choosing 
the value of k to use in making predictions. 
Here, we use a validation set to 
choose this value. Write a loop 
that does the following:

   For k in [1, 2, ..., 15]:
      Makes predictions for each house in the VALIDATION set using the k-nearest neighbors from the TRAINING set.
      Computes the RSS for these predictions on the VALIDATION set
      Stores the RSS computed above in rss_all
   Report which k produced the lowest RSS on VALIDATION set.

(Depending on your computing environment, 
this computation may take 10-15 minutes.)
"""
# something is wrong here.
def choose_best_k(features_train, output_train, features_valid, output_valid, max_k):
    predictions_matrix = np.zeros((len(output_valid), max_k+1))
#    print "np.shape(output_valid) =", np.shape(output_valid) (1435,)
    for k in range(1, max_k+1):
        v = find_predictions_for_a_set(k, 
                                       features_train, 
                                       output_train, 
                                       features_valid)
        predictions_matrix[:, k] = v.reshape(len(v),)
    output_valid = output_valid.reshape(len(output_valid), 1)
    residuals_squared = (predictions_matrix - output_valid) ** 2
    RSS_valiation_set = np.sum(residuals_squared, axis=0)
    print "RSS_valiation_set =", RSS_valiation_set
    return np.argmin(RSS_valiation_set)

# find best k, using validation set.
print "best k is ", choose_best_k(features_train, output_train, features_valid, output_valid, 15)
print "--------------------------------------"







