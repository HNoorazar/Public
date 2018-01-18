import graphlab
import numpy as np
from math import log
from math import sqrt
import matplotlib.pyplot as plt
# Load house sales data
sales = graphlab.SFrame('kc_house_data.gl/')

# Split data into training and testing
train_data,test_data = sales.random_split(.8,seed=0)


"""
we're going to write a polynomial function 
that takes an SArray and a maximal degree 
and returns an SFrame with columns containing 
the SArray to all the powers up to 
the maximal degree.

The easiest way to apply a power 
to an SArray is to use the .apply() 
and lambda x: functions. For example 
to take the example array and compute 
the third power we can do as follows: 
(note running this cell the first time 
may take longer than expected since it 
loads graphlab), e.g.

tmp = graphlab.SArray([1., 2., 3.])
tmp_cubed = tmp.apply(lambda x: x**3)
print tmp
print tmp_cubed
"""

############### Create SFrame columns.
"""
We can create an empty SFrame 
using graphlab.SFrame() and 
then add any columns to it with 
ex_sframe['column_name'] = value. 

For example we create an empty 
SFrame and make the column 'power_1' 
to be the first power of tmp 
(i.e. tmp itself). e.g.

ex_sframe = graphlab.SFrame()
ex_sframe['power_1'] = tmp
print ex_sframe
"""


"""
# Polynomial_sframe function

Using the hints above complete 
the following function to create 
an SFrame consisting of the powers 
of an SArray up to a specific degree:
"""

def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature.apply(lambda x: x**power)
    return poly_sframe

# to test your function 
# consider the smaller tmp 
# variable and what you 
# would expect the outcome 
# of the following call:
tmp = graphlab.SArray([1., 2., 3.])
# print polynomial_sframe(tmp, 3)


"""
Visualizing polynomial regression

Let's use matplotlib to visualize 
what a polynomial regression looks 
like on some real data.
"""
##########
"""
As in Week 3, we will use 
the sqft_living variable. 
For plotting purposes 
(connecting the dots), you'll need 
to sort by the values of sqft_living. 
For houses with identical square footage, 
we break the tie by their prices.
"""
sales = sales.sort(['sqft_living', 'price'])
"""
# Let's start with a degree 1 polynomial 
# using 'sqft_living' (i.e. a line) to 
# predict 'price' and plot what it looks like.
"""
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target
"""
# NOTE: for all the models in 
# this notebook use validation_set = None 
# to ensure that all results are 
# consistent across users.
"""
model1 = graphlab.linear_regression.create(poly1_data,
                                           target = 'price',
                                           features = ['power_1'],
                                           validation_set = None,
                                           verbose = False)

"""
let's take a look at the weights before we plot
"""
# print model1.get("coefficients")

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')

"""
Let's unpack that plt.plot() command. 
The first pair of SArrays we passed are 
the 1st power of sqft and the actual 
price we then ask it to print these 
as dots '.'. 
The next pair we pass is the 1st power 
of sqft and the predicted values from 
the linear model. We ask these to be 
plotted as a line '-'.
We can see, not surprisingly, that 
the predicted values all fall on a 
line, specifically the one with 
slope 280 and intercept -43579. 
What if we wanted to plot a second degree polynomial?
"""
poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data,
                                           target = 'price',
                                           features = my_features,
                                           validation_set = None,
                                           verbose = False)

# print model2.get("coefficients")
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')


"""
Degree 15
"""

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = graphlab.linear_regression.create(poly15_data, 
                                            target = 'price', 
                                            features = my_features, 
                                            validation_set = None,
                                            verbose = False)

# print model15.get("coefficients")

"""
Changing the data and re-learning

We're going to split the sales data 
into four subsets of roughly equal size. 

Then you will estimate a 15th degree 
polynomial model on all four subsets 
of the data. Print the coefficients 
(you should use .print_rows(num_rows = 16) 
to view all of them) and plot the 
resulting fit (as we did above). 

The quiz will ask you some questions 
about these results.

To split the sales data into four subsets, 
we perform the following steps:
First split sales into 2 subsets with
 .random_split(0.5, seed=0).
Next split the resulting subsets 
into 2 more subsets each. 
Use .random_split(0.5, seed=0).
We set seed=0 in these steps so 
that different users get consistent 
results. You should end up with 4 
subsets (set_1, set_2, set_3, set_4) 
of approximately equal size.
"""
bigSet_1, bigSet_2 = sales.random_split(.5, seed=0)
set_1, set_2 = bigSet_1.random_split(.5, seed=0)
set_3, set_4 = bigSet_2.random_split(.5, seed=0)

"""
Fit a 15th degree polynomial 
on set_1, set_2, set_3, and set_4 
using sqft_living to predict prices. 

Print the coefficients and make 
a plot of the resulting model.
"""
# set_1 model:
poly15_data_1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_data_1.column_names() # get the name of the features
poly15_data_1['price'] = set_1['price'] # add price to the data since it's the target
set_1_model15 = graphlab.linear_regression.create(poly15_data_1, 
                                                  target = 'price', 
                                                  features = my_features, 
                                                  validation_set = None,
                                                  verbose = False)
print "set_1_coefficient:" 
A = set_1_model15.get("coefficients")
A.print_rows(num_rows = 16)

# set_2 model:
poly15_data_2 = polynomial_sframe(set_2['sqft_living'], 15)
# my_features = poly15_data_2.column_names() # get the name of the features
poly15_data_2['price'] = set_2['price'] # add price to the data since it's the target
set_2_model15 = graphlab.linear_regression.create(poly15_data_2, 
                                                  target = 'price', 
                                                  features = my_features, 
                                                  validation_set = None,
                                                  verbose = False)
print "set_2_coefficient:" 
A = set_2_model15.get("coefficients")
A.print_rows(num_rows = 16)

# set_3 model:
poly15_data_3 = polynomial_sframe(set_3['sqft_living'], 15)
# my_features = poly15_data_2.column_names() # get the name of the features
poly15_data_3['price'] = set_3['price'] # add price to the data since it's the target
set_3_model15 = graphlab.linear_regression.create(poly15_data_3, 
                                                  target = 'price', 
                                                  features = my_features, 
                                                  validation_set = None,
                                                  verbose = False)
print "set_3_coefficient:"
A = set_3_model15.get("coefficients")
A.print_rows(num_rows = 16)

# set_4 model:
poly15_data_4 = polynomial_sframe(set_4['sqft_living'], 15)
# my_features = poly15_data_4.column_names() # get the name of the features
poly15_data_4['price'] = set_4['price'] # add price to the data since it's the target
set_4_model15 = graphlab.linear_regression.create(poly15_data_4, 
                                                  target = 'price', 
                                                  features = my_features, 
                                                  validation_set = None,
                                                  verbose = False)
print "set_4_coefficient:" 
A = set_4_model15.get("coefficients")
A.print_rows(num_rows = 16)



"""
Selecting a Polynomial Degree

Whenever we have a "magic" parameter 
like the degree of the polynomial 
there is one well-known way to select 
these parameters: validation set. 
(We will explore another approach in week 4).

We split the sales dataset 3-way 
into training set, test set, and 
validation set as follows:
Split our sales data into 2 sets: 
training_and_validation and testing. 
Use random_split(0.9, seed=1).

Further split our training data 
into two sets: training and validation. 
Use random_split(0.5, seed=1).
Again, we set seed=1 to obtain 
consistent results for different users.
"""

training_and_validation, test_data = sales.random_split(0.9, seed=1)
training_data, validation_data = training_and_validation.random_split(.5, seed=1)

"""
Next you should write a loop that does the following:

** For degree in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 
(to get this in python type range(1, 15+1))
  1- Build an SFrame of polynomial data of 
  train_data['sqft_living'] at the current degree

  2- hint: my_features = poly_data.column_names() 
  gives you a list e.g. ['power_1', 'power_2', 'power_3'] 
  which you might find useful for 
  graphlab.linear_regression.create( features = my_features)

  3- Add train_data['price'] to the polynomial SFrame

  4- Learn a polynomial regression model 
  to sqft vs price with that degree on TRAIN data
  
  5- Compute the RSS on VALIDATION data 
  (here you will want to use .predict()) 
  for that degree and you will need to 
  make a polynmial SFrame using validation data.

** Report which degree had the lowest RSS 
on validation data (remember python indexes from 0)
(Note you can turn off the print out of 
linear_regression.create() with verbose = False)
"""

poly_degree = range(1, 15+1)
RSS_vector = np.zeros((15,1))

for degree_count in poly_degree:
    poly_training_data = polynomial_sframe(training_data['sqft_living'], degree_count)
    my_features = poly_training_data.column_names()
    poly_training_data['price'] = training_data['price'] # add price to the data since it's the target
    model = graphlab.linear_regression.create(poly_training_data, 
                                              target = 'price', 
                                              features = my_features, 
                                              validation_set = None,
                                              verbose = False)
    poly_validation_data = polynomial_sframe(validation_data['sqft_living'], degree_count)
    validation_predictions = model.predict(poly_validation_data)
    validation_residuals = validation_predictions - validation_data['price']
    validation_residuals_squared = validation_residuals ** 2
    RSS_vector[degree_count-1] = validation_residuals_squared.sum()

# print "RSS_vector=", RSS_vector
poly6_training_data = polynomial_sframe(training_data['sqft_living'], 6)
my_features = poly6_training_data.column_names()
poly6_training_data['price'] = training_data['price'] # add price to the data since it's the target
model6 = graphlab.linear_regression.create(poly6_training_data, 
                                          target = 'price', 
                                          features = my_features, 
                                          validation_set = None,
                                          verbose = False)
poly6_test_data = polynomial_sframe(test_data['sqft_living'], 6)
test_predictions = model6.predict(poly6_test_data)
test_residuals = test_predictions - test_data['price']
test_residuals_squared = test_residuals ** 2
RSS_test = test_residuals_squared.sum()
print "RSS_test=", RSS_test
