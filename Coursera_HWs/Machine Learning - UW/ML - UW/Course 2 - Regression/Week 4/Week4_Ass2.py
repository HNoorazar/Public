"""
Regression. UW. Week 4. Assignment 2.
"""

import numpy as np
import graphlab
import matplotlib.pyplot as plt

sales = graphlab.SFrame('kc_house_data.gl/')

"""
Regression Week 4: Ridge Regression (gradient descent)

In this notebook, you will 
implement ridge regression 
via gradient descent. You will:
Convert an SFrame into a Numpy array
Write a Numpy function to 
compute the derivative of 
the regression weights with 
respect to a single feature

Write gradient descent function 
to compute the regression weights 
given an initial weight vector, 
step size, tolerance, and L2 penalty

If we want to do any "feature engineering" 
like creating new features or adjusting 
existing ones we should do this directly 
using the SFrames as seen in the first 
notebook of Week 2. For this notebook, 
however, we will work with the existing 
features.

# Import useful functions from previous notebook

As in Week 2, we convert the SFrame 
into a 2D Numpy array. Copy and 
paste get_numpy_data() from the 
second notebook of Week 2.
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
Computing the Derivative
We are now going to move to computing 
the derivative of the regression cost function. 
Recall that the cost function is the sum over 
the data points of the squared difference 
between an observed output and a predicted 
output, plus the L2 penalty term.

Cost(w) = SUM[ (prediction - output)^2 ]+ l2_penalty*(w[0]^2 + w[1]^2 + ... + w[k]^2).

Since the derivative of a sum is 
the sum of the derivatives, we can 
take the derivative of the first 
part (the RSS) as we did in the 
notebook for the unregularized 
case in Week 2 and add the derivative 
of the regularization part. As we saw, 
the derivative of the RSS with 
respect to w[i] can be written as:

2*SUM[ error*[feature_i] ].

The derivative of the regularization 
term with respect to w[i] is:

2*l2_penalty*w[i].

Summing both, we get 2*SUM[ error*[feature_i] ] + 2*l2_penalty*w[i].

That is, the derivative for 
the weight for feature i is 
the sum (over data points) 
of 2 times the product of 
the error and the feature 
itself, plus 2*l2_penalty*w[i].

We will not regularize the 
constant. Thus, in the case 
of the constant, the derivative 
is just twice the sum of the 
errors (without the 2*l2_penalty*w[0] term).

Recall that twice the sum of 
the product of two vectors is 
just twice the dot product of 
the two vectors. 

Therefore the derivative for 
the weight for feature_i is 
just two times the dot product 
between the values of feature_i 
and the current errors, plus 
2*l2_penalty*w[i]. With this in mind 
complete the following derivative 
function which computes the 
derivative of the weight given 
the value of the feature 
(over all data points) and 
the errors (over all data points). 

To decide when to we are dealing 
with the constant (so we don't regularize it) 
we added the extra parameter to 
the call feature_is_constant which 
you should set to True when computing 
the derivative of the constant and 
False otherwise.
"""
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if feature_is_constant == True:
        derivative = 2 * np.dot(errors, feature)
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
        derivative = 2 * np.dot(errors, feature) + (2 * l2_penalty * weight)
    return derivative
"""
## To test your feature derivartive run the following:
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights) 
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
print np.sum(errors*example_features[:,1])*2+20.
print ''

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
print np.sum(errors)*2.
"""


"""
### Gradient Descent

With this in mind, complete 
the following gradient descent 
function below using your 
derivative function above. 
For each step in the gradient descent, 
we update the weight for 
each feature before computing 
our stopping criteria.
"""
def ridge_regression_gradient_descent(feature_matrix, 
                                      output, 
                                      initial_weights, 
                                      step_size, 
                                      l2_penalty, 
                                      max_iterations=100):
    print 'Starting gradient descent with l2_penalty = ' + str(l2_penalty)
    
    weights = np.array(initial_weights) # make sure it's a numpy array
    iteration = 0 # iteration counter
    print_frequency = 1  # for adjusting frequency of debugging output
    
    #while not reached maximum number of iterations:
    while iteration < max_iterations:
        iteration += 1  # increment iteration counter
        """
        ### === code section for adjusting frequency of debugging output. ===
        if iteration == 10:
            print_frequency = 10
        if iteration == 100:
            print_frequency = 100
        if iteration%print_frequency==0:
            print('Iteration = ' + str(iteration))
        ### === end code section ===
        """
        
        # compute the predictions based on 
        # feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)

        # compute the errors as predictions - output
        errors = predictions - output
        
        # from time to time, print the value of the cost function
        # if iteration%print_frequency==0:
          #  print 'Cost function = ', str(np.dot(errors,errors) + l2_penalty*(np.dot(weights,weights) - weights[0]**2))
        
        for i in xrange(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            # (Remember: when i=0, you are computing the derivative of the constant!)
            if i==0:
            # Why are we even doing this?
                feature_is_constant = True
                deriv = feature_derivative_ridge(errors, 
                                                 feature_matrix[:,0], 
                                                 weights[0], 
                                                 l2_penalty, 
                                                 feature_is_constant)
                # weights[0] = weights[0] - deriv * step_size
            else:
                feature_is_constant = False
                deriv = feature_derivative_ridge(errors, 
                                                 feature_matrix[:,i], 
                                                 weights[i], 
                                                 l2_penalty, 
                                                 feature_is_constant)
                weights[i] = weights[i] - deriv * step_size

            # subtract the step size times the derivative from the current weight
            
    print 'Done with gradient descent at iteration ', iteration
    print 'Learned weights = ', str(weights)
    return weights


simple_features = ['sqft_living']
my_output = 'price'
train_data,test_data = sales.random_split(.8,seed=0)
"""
In this part, we will only 
use 'sqft_living' to predict 'price'. 
Use the get_numpy_data 
function to get a Numpy 
versions of your data with 
only this feature, for both 
the train_data and the test_data.
"""
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

"""
Let's set the parameters for our optimization:
"""
initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000

"""
First, let's consider no regularization. 
Set the l2_penalty to 0.0 
and run your ridge regression 
algorithm to learn the weights 
of your model. 
Call your weights:
simple_weights_0_penalty
we'll use them later.
"""
l2_penalty = 0.0 
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, 
                                                             output, 
                                                             initial_weights, 
                                                             step_size, 
                                                             l2_penalty, 
                                                             max_iterations)
# print "simple_weights_0_penalty = ", simple_weights_0_penalty
"""
Next, let's consider high regularization. 
Set the l2_penalty to 1e11 and 
run your ridge regression algorithm 
to learn the weights of your model. 
Call your weights:
simple_weights_high_penalty
"""
l2_penalty = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, 
                                                                output, 
                                                                initial_weights, 
                                                                step_size, 
                                                                l2_penalty, 
                                                                max_iterations)
# print "simple_weights_high_penalty =", simple_weights_high_penalty
"""
This code will plot the two 
learned models. 
(The blue line is for the model 
with no regularization and the 
red line is for the one with 
high regularization.)
"""

plt.plot(simple_feature_matrix,output,'k.',
          simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
          simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')

(simple_feature_matrix, output) = get_numpy_data(test_data, simple_features, my_output)


weights = np.array([0., 0.])
predictions_initials = np.dot(simple_feature_matrix, weights)
RSS_initial = np.sum((predictions_initials - output) ** 2)
print "RSS_initial=", RSS_initial

weights = simple_weights_0_penalty
predictions_simple = np.dot(simple_feature_matrix, weights)
RSS_simple = np.sum((predictions_simple - output) ** 2)
print "RSS_simple=", RSS_simple

weights = simple_weights_high_penalty
predictions_high = np.dot(simple_feature_matrix, weights)
RSS_high = np.sum((predictions_high - output) ** 2)
print "RSS_high=", RSS_high


"""
Running a multiple regression with L2 penalty
Let us now consider a model with 2 features: ['sqft_living', 'sqft_living15'].
First, create Numpy versions of your training and test data with these two features.
"""
model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
"""
We need to re-inialize the weights, 
since we have one extra parameter. 
Let us also set the step size 
and maximum number of iterations.
"""
initial_weights = np.array([0.0, 0.0, 0.0])
step_size = 1e-12
max_iterations = 1000


"""
First, let's consider no regularization. 
Set the l2_penalty to 0.0 and 
run your ridge regression algorithm 
to learn the weights of your model. 
Call your weights:
multiple_weights_0_penalty
"""
l2_penalty = 0.0
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, 
                                                               output, 
                                                               initial_weights, 
                                                               step_size, 
                                                               l2_penalty, 
                                                               max_iterations)
print " multiple_weights_0_penalty = ", multiple_weights_0_penalty
"""
Next, let's consider high regularization. 
Set the l2_penalty to 1e11 and 
run your ridge regression algorithm 
to learn the weights of your model. 
Call your weights:
multiple_weights_high_penalty
"""
l2_penalty = 1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, 
                                                                  output, 
                                                                  initial_weights, 
                                                                  step_size, 
                                                                  l2_penalty, 
                                                                  max_iterations)
weights = np.array([0., 0., 0.])
predictions_initials = np.dot(test_feature_matrix, weights)
RSS_initial = np.sum((predictions_initials - test_output) ** 2)
print "RSS_initial=", RSS_initial

weights = multiple_weights_0_penalty
predictions_no_reg = np.dot(test_feature_matrix, weights)
RSS_simple = np.sum((predictions_no_reg - test_output) ** 2)
print "RSS_simple=", RSS_simple

weights = multiple_weights_high_penalty
predictions_high = np.dot(test_feature_matrix, weights)
RSS_high = np.sum((predictions_high - test_output) ** 2)
print "RSS_high=", RSS_high

"""
Predict the house price 
for the 1st house in the 
test set using the no regularization 
and high regularization models. 
(Remember that python starts indexing from 0.) 
How far is the prediction from 
the actual price? Which weights 
perform best for the 1st house?
"""
first_hourse_prediction_no_regularization = predictions_simple[0]
print "first_hourse_prediction_no_regularization = ", first_hourse_prediction_no_regularization
print "first_error_no_penalty = ", test_output[0] - first_hourse_prediction_no_regularization


first_hourse_prediction_high_regularization = predictions_high[0]
print "first_hourse_prediction_high_regularization = ", first_hourse_prediction_high_regularization
print "first_error_high_penalty = ", test_output[0] - first_hourse_prediction_high_regularization