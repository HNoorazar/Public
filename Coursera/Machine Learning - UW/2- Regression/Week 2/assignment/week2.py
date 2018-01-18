import graphlab
import numpy as np
from math import log
from math import sqrt
# Load house sales data
sales = graphlab.SFrame('kc_house_data.gl/')

# Split data into training and testing
train_data,test_data = sales.random_split(.8,seed=0)

"""
Recall we can use the following code to learn
a multiple regression model predicting 'price'
based on the following features:
example_features = ['sqft_living', 'bedrooms', 'bathrooms'] 
on training data with the following code:
(Aside: We set validation_set = None to ensure that the results are always the same)
"""

example_features = ['sqft_living', 'bedrooms', 'bathrooms']
example_model = graphlab.linear_regression.create(train_data,
                                                  target = 'price',
                                                  features = example_features,
                                                  validation_set = None)

# Extract coefficients
example_weight_summary = example_model.get("coefficients")
print "Here=", example_weight_summary

# prediction
example_predictions = example_model.predict(train_data)
print example_predictions[0] # should be 271789.505878

# Compute RSS
def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = model.predict(data)
    # Then compute the residuals/errors
    residuals = outcome - predictions
    # Then square and add them up
    RSS = (residuals*residuals).sum()
    return(RSS)    
    

# Test your function by computing
# the RSS on TEST data for the example model:
rss_example_train = get_residual_sum_of_squares(example_model, test_data, test_data['price'])
print rss_example_train # should be 2.7376153833e+14

"""
Create some new features

Although we often think of multiple regression
as including multiple different features 
(e.g. # of bedrooms, squarefeet, and # of bathrooms) 
but we can also consider transformations of existing 
features e.g. the log of the squarefeet or even 
"interaction" features such as the product of 
bedrooms and bathrooms.
"""

# Next create the following 4 new features as column in both TEST and TRAIN data:
train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)
print "bedrooms_squared_mean = ", test_data['bedrooms_squared'].mean()

train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
test_data['bed_bath_rooms']  = test_data['bedrooms']  * test_data['bathrooms']
print "bed_bath_rooms_mean", test_data['bed_bath_rooms'].mean()

train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living']  = test_data['sqft_living'].apply(lambda x: log(x))
print "log_sqft_living_mean", test_data['log_sqft_living'].mean()

train_data['lat_plus_long'] = train_data['lat'] + train_data['long']
test_data['lat_plus_long'] =  test_data['lat']  + test_data['long']
print "lat_plus_long_mean", test_data['lat_plus_long'].mean()


"""
Learning Multiple Models
Now we will learn the weights for three (nested) 
models for predicting house prices. The first 
model will have the fewest features the second 
model will add one more feature and the third 
will add a few more:

Model 1: squarefeet, # bedrooms, # bathrooms, latitude & longitude
Model 2: add bedrooms*bathrooms
Model 3: Add log squarefeet, bedrooms squared, and the (nonsensical) latitude + longitude
"""
model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

model1 = graphlab.linear_regression.create(train_data, target = 'price', 
                                           features = model_1_features, 
                                           validation_set = None)

model2 = graphlab.linear_regression.create(train_data, target = 'price',
                                           features = model_2_features, 
                                           validation_set = None)

model3 = graphlab.linear_regression.create(train_data, target = 'price',
                                           features = model_3_features, 
                                           validation_set = None)

model1_summary = model1.get("coefficients")
print "model1_summary= ", model1_summary
model2_summary = model2.get("coefficients")
print "model2_summary= ", model2_summary
model3_summary = model3.get("coefficients")
print "model3_summary= ", model3_summary

"""
Comparing multiple models
Now that you've learned three models 
and extracted the model weights we 
want to evaluate which model is best.

First use your functions from earlier 
to compute the RSS on TRAINING Data 
for each of the three models.
"""

RSS1_train = get_residual_sum_of_squares(model1, train_data, train_data['price'])
RSS2_train = get_residual_sum_of_squares(model2, train_data, train_data['price'])
RSS3_train = get_residual_sum_of_squares(model3, train_data, train_data['price'])
# print "Training RSS=", RSS1_train, RSS2_train, RSS3_train

RSS1_test = get_residual_sum_of_squares(model1, test_data, test_data['price'])
RSS2_test = get_residual_sum_of_squares(model2, test_data, test_data['price'])
RSS3_test = get_residual_sum_of_squares(model3, test_data, test_data['price'])
# print "Test RSS=", RSS1_test, RSS2_test, RSS3_test

############################################
############################################
"""
############################################    Assignment 2 of Week 2
"""
############################################
############################################

"""
Multiple Regression (gradient descent)

In the first notebook we explored 
multiple regression using graphlab 
create. Now we will use graphlab 
along with numpy to solve for the 
regression weights with gradient descent.

In this notebook we will cover 
estimating multiple regression 
weights via gradient descent. 
You will:

1- Add a constant column of 1's to a graphlab SFrame to account for the intercept
2- Convert an SFrame into a Numpy array
3- Write a predict_output() function using Numpy

4- Write a numpy function to compute 
the derivative of the regression 
weights with respect to a single feature

5- Write gradient descent function 
to compute the regression weights 
given an initial weight vector, 
step size and tolerance.

6- Use the gradient descent function to estimate regression weights for multiple features
"""

"""
Convert to Numpy Array

Although SFrames offer a number 
of benefits to users (especially 
when using Big Data and built-in 
graphlab functions) in order to 
understand the details of the 
implementation of algorithms 
it's important to work with a 
library that allows for direct 
(and optimized) matrix operations. 
Numpy is a Python solution to work 
with matrices (or any multi-dimensional "array").

Recall that the predicted value 
given the weights and the features 
is just the dot product between the 
feature and weight vector. 
Similarly, if we put all of the 
features row-by-row in a matrix then 
the predicted value for all the observations 
can be computed by right multiplying the 
"feature matrix" by the "weight vector".
First we need to take the SFrame of 
our data and convert it into a 2D 
numpy array (also called a matrix). 
To do this we use graphlab's 
built in .to_dataframe() which 
converts the SFrame into a Pandas 
(another python library) dataframe. 
We can then use Panda's .as_matrix() 
to convert the dataframe into a numpy matrix.
"""

"""
Now we will write a function that 
will accept an SFrame, a list of 
feature names (e.g. ['sqft_living', 'bedrooms']) 
and an target feature e.g. ('price') 
and will return two things:
1 - A numpy matrix whose columns are the 
desired features plus a constant column (this is how we create an 'intercept')

2- A numpy array containing the values of the output
With this in mind, complete the following 
function (where there's an empty line you 
should write a line of code that does what 
the comment above indicates)

Please note you will need GraphLab Create 
version at least 1.7.1 in order for .to_numpy() to work!
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

"""
For testing let's use the 'sqft_living' 
feature and a constant as our features 
and price as our output:
"""
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
# the [] around 'sqft_living' makes it a list
print "example_features[0,:]= ", example_features[0,:] 
print "example_output[0]= ", example_output[0]


"""
Predicting output given regression weights

Suppose we had the weights [1.0, 1.0] and 
the features [1.0, 1180.0] and we wanted 
to compute the predicted output 
1.0*1.0 + 1.0*1180.0 = 1181.0 this is 
the dot product between these two arrays. 
If they're numpy arrayws we can use np.dot() to compute this:
"""

my_weights = np.array([1., 1.]) # the example weights
my_features = example_features[0,] # we'll use the first data point
predicted_value = np.dot(my_features, my_weights)
print "line273: predicted value", predicted_value


def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)
"""
# Test
test_predictions = predict_output(example_features, my_weights)
print test_predictions[0] # should be 1181.0
print test_predictions[1] # should be 2571.0
"""
###########################################################################
"""
Computing the Derivative

We are now going to move to computing 
the derivative of the regression cost 
function. Recall that the cost function
is the sum over the data points of the 
squared difference between an observed 
output and a predicted output.
Since the derivative of a sum is the sum 
of the derivatives we can compute the 
derivative for a single data point and 
then sum over data points. We can write 
the squared difference between the observed 
output and predicted output for a single 
point as follows:

(w[0]*[CONSTANT] + w[1]*[feature_1] +...+ w[i]*[feature_i] +...+ w[k]*[feature_k]-output)^2

Where we have k features and a constant. 
So the derivative with respect to weight 
w[i] by the chain rule is:

2*(w[0]*[CONSTANT] + w[1]*[feature_1] +...+ w[i]*[feature_i] +...+ w[k]*[feature_k] - output)*[feature_i]

The term inside the paranethesis is 
just the error (difference between 
prediction and output). So we can re-write this as:

2*error*[feature_i]

That is, the derivative for the 
weight for feature i is the sum 
(over data points) of 2 times the 
product of the error and the feature 
itself. In the case of the 
constant then this is just twice 
the sum of the errors!

Recall that twice the sum of 
the product of two vectors is 
just twice the dot product of 
the two vectors. Therefore the 
derivative for the weight for 
feature_i is just two times 
the dot product between the 
values of feature_i and the 
current errors.

With this in mind complete the 
following derivative function 
which computes the derivative of 
the weight given the value of the 
feature (over all data points) and 
the errors (over all data points).
"""
def feature_derivative(errors, feature):
    # Assume that errors and feature are both 
    # numpy arrays of the same length (number of data points)
    
    # compute twice the dot product of these 
    # vectors as 'derivative' and return the value
    derivative = 2 * np.dot(errors, feature)
    return(derivative)

""" 
Test
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
my_weights = np.array([0., 0.]) # this makes all the predictions 0
test_predictions = predict_output(example_features, my_weights) 
# just like SFrames 2 numpy arrays can be elementwise subtracted with '-': 
errors = test_predictions - example_output # prediction errors in this case is just the -example_output
feature = example_features[:,0] # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
derivative = feature_derivative(errors, feature)
print "derivative=", derivative
print -np.sum(example_output)*2 # should be the same as derivative
"""
##########################################################################################
"""
Gradient Descent

Now we will write a function that 
performs a gradient descent. The basic 
premise is simple. Given a starting 
point we update the current weights 
by moving in the negative gradient direction. 

Recall that the gradient is the direction 
of increase and therefore the negative 
gradient is the direction of decrease 
and we're trying to minimize a cost function.

The amount by which we move in the negative 
gradient direction is called the 'step size'. 
We stop when we are 'sufficiently close' to 
the optimum. 

We define this by requiring that 
the magnitude (length) of the gradient vector 
to be smaller than a fixed 'tolerance'.
With this in mind, complete the following gradient 
descent function below using your derivative function above. 
For each step in the gradient descent we update the
weight for each feature befofe computing 
our stopping criteria
"""
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array
    while not converged:
        # compute the predictions based on feature_matrix 
        # and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)

        # compute the errors as predictions - output
        residuals = predictions - output
        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, 
        # update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the 
            # feature column associated with weights[i]
            # compute the derivative for weight[i]:
            deriv = feature_derivative(residuals, feature_matrix[:,i])
            # add the squared value of the derivative 
            # to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares += (deriv ** 2)
            # subtract the step size times the derivative from the current weight
            weights[i] -= step_size * deriv
        # compute the square-root of the gradient 
        # sum of squares to get the gradient magnitude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)
    
"""
A few things to note before we 
run the gradient descent. 
Since the gradient is a sum 
over all the data points and 
involves a product of an error 
and a feature the gradient itself 
will be very large since the 
features are large (squarefeet) 
and the output is large (prices). 

So while you might expect "tolerance" 
to be small, small is only relative 
to the size of the features.
For similar reasons the step size 
will be much smaller than you might 
expect but this is because the 
gradient has such large values.
"""
########################################################################################
"""
Running the Gradient Descent as Simple Regression
"""

"""
Although the gradient descent is designed 
for multiple regression since the constant 
is now a feature we can use the gradient 
descent function to estimat the parameters 
in the simple regression on squarefeet. 

The folowing cell sets up the feature_matrix, 
output, initial weights and step size for the first model:
"""

# let's test out the gradient descent
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7
new_weights1 = regression_gradient_descent(simple_feature_matrix,
                                          output, initial_weights,
                                          step_size, tolerance)
print "Quiz Question 1: ","new_weights1=", new_weights1
print "---------------------------------------------------------"

(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
test_predictions = predict_output(test_simple_feature_matrix, new_weights1)

print "Quiz Question 2"
print "test_predictions[0]=", test_predictions[0]
print "---------------------------------------------------------"

test_residuals = test_output - test_predictions
test_RSS1 = (test_residuals * test_residuals).sum()
print "test_RSS1=", test_RSS1


"""
Running a multiple regression

Now we will use more than one 
actual feature. 

Use the following code to produce 
the weights for a second model 
with the following parameters:

Use the below parameters to estimate the model weights. 
Record these values for your quiz.
"""

model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

new_weights2 = regression_gradient_descent(feature_matrix,
                                          output, initial_weights,
                                          step_size, tolerance)
                                          
print "new_weights2 =", new_weights2
print "---------------------------------------------------------"
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

test_predictions2 = predict_output(test_feature_matrix, new_weights2)
print "Quiz Question 3: ", "predictions2[0]=", test_predictions2[0]
print "---------------------------------------------------------"

print "actial price of first house in the test data=", test_output[0]
print "---------------------------------------------------------"
test_residuals2 = test_output - test_predictions2
test_RSS2 = (test_residuals2 * test_residuals2).sum()
print "test_RSS2=", test_RSS2
print "---------------------------------------------------------"