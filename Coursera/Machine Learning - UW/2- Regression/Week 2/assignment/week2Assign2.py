import graphlab
import numpy as np
sales = graphlab.SFrame('kc_house_data.gl/')

############################################
############################################
"""
Assignment 2 of Week 2
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
#print example_features[0,:] 
#print example_output[0]


"""
Predicting output given regression weights
Suppose we had the weights [1.0, 1.0] and 
the features [1.0, 1180.0] and we wanted 
to compute the predicted output 
1.0*1.0 + 1.0*1180.0 = 1181.0 this is 
the dot product between these two arrays. 
If they're numpy arrayws we can use np.dot() to compute this:
"""
