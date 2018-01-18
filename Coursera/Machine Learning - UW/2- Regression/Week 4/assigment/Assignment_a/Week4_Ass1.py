"""
Regression. UW. Week 4. Assignment 1.
"""
import graphlab
import numpy as np
import matplotlib.pyplot as plt


sales = graphlab.SFrame('kc_house_data.gl/')
sales = sales.sort(['sqft_living','price'])

"""
Polynomial regression, revisited

We build on the material from Week 3, 
where we wrote the function to produce 
an SFrame with columns containing the 
powers of a given input. Copy and paste 
the function polynomial_sframe from 
Week 3:
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

"""
Let us revisit the 15th-order 
polynomial model using the 
'sqft_living' input. 

Generate polynomial features up to 
degree 15 using polynomial_sframe() 
and fit a model with these features. 
When fitting the model, use an L2 penalty of 1e-5:

Note: When we have so many features 
and so few data points, the solution 
can become highly numerically unstable, 
which can sometimes lead to strange 
unpredictable results. Thus, rather 
than using no regularization, we will 
introduce a tiny amount of regularization 
(l2_penalty=1e-5) to make the solution 
numerically stable. (In lecture, we discussed 
the fact that regularization can also 
help with numerical stability, and 
here we are seeing a practical example.)
With the L2 penalty specified above, 
fit the model and print out the 
learned weights.
Hint: make sure to add 'price' column to 
the new SFrame before calling 
graphlab.linear_regression.create(). 
Also, make sure GraphLab Create doesn't 
create its own validation set by 
using the option validation_set=None 
in this call.

"""
l2_small_penalty = 1e-5
poly_degree = range(1, 15+1)

for degree_count in poly_degree:
    poly_training_data = polynomial_sframe(sales['sqft_living'], degree_count)
    my_features = poly_training_data.column_names()
    poly_training_data['price'] = sales['price'] # add price to the data since it's the target
    model = graphlab.linear_regression.create(poly_training_data, 
                                              target = 'price', 
                                              features = my_features,
                                              l2_penalty= l2_small_penalty,
                                              validation_set = None,
                                              verbose = False)
    if degree_count == 15:
        print "Question 1:" 
        print "######################"
        A = model.get("coefficients")
        A.print_rows(num_rows = 2)
        print "######################"

"""
## Observe overfitting

Recall from Week 3 that the 
polynomial fit of degree 15 
changed wildly whenever the 
data changed. In particular, 
when we split the sales data 
into four subsets and fit 
the model of degree 15, the 
result came out to be very 
different for each subset. 
The model had a high variance. 
We will see in a moment that 
ridge regression reduces such 
variance. But first, we must 
reproduce the experiment we 
did in Week 3.
First, split the data into 
split the sales data into 
four subsets of roughly equal 
size and call them set_1, set_2, 
set_3, and set_4. 
Use .random_split function and 
make sure you set seed=0.
"""        
(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

"""
Next, fit a 15th degree polynomial 
on set_1, set_2, set_3, and set_4, 
using 'sqft_living' to predict prices. 
Print the weights and make a plot 
of the resulting model.
Hint: When calling graphlab.linear_regression.create(), 
use the same L2 penalty as 
before (i.e. l2_small_penalty).

Also, make sure GraphLab Create 
doesn't create its own validation 
set by using the option 
validation_set = None in this call.
"""
# set_1 model:
poly15_data_1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_data_1.column_names() # get the name of the features
poly15_data_1['price'] = set_1['price'] # add price to the data since it's the target
set_1_model15 = graphlab.linear_regression.create(poly15_data_1, 
                                                  target = 'price', 
                                                  features = my_features,
                                                  l2_penalty= l2_small_penalty,
                                                  validation_set = None,
                                                  verbose = False)
print "############################################"
print "set_1_coefficient:" 
A = set_1_model15.get("coefficients")
A.print_rows(num_rows = 2)
# set_2 model:
poly15_data_2 = polynomial_sframe(set_2['sqft_living'], 15)
# my_features = poly15_data_2.column_names() # get the name of the features
poly15_data_2['price'] = set_2['price'] # add price to the data since it's the target
set_2_model15 = graphlab.linear_regression.create(poly15_data_2, 
                                                  target = 'price', 
                                                  features = my_features, 
                                                  l2_penalty= l2_small_penalty,
                                                  validation_set = None,
                                                  verbose = False)
print "############################################"
print "set_2_coefficient:" 
A = set_2_model15.get("coefficients")
A.print_rows(num_rows = 2)
# set_3 model:
poly15_data_3 = polynomial_sframe(set_3['sqft_living'], 15)
# my_features = poly15_data_2.column_names() # get the name of the features
poly15_data_3['price'] = set_3['price'] # add price to the data since it's the target
set_3_model15 = graphlab.linear_regression.create(poly15_data_3, 
                                                  target = 'price', 
                                                  features = my_features, 
                                                  l2_penalty= l2_small_penalty,
                                                  validation_set = None,
                                                  verbose = False)
print "############################################"
print "set_3_coefficient:"
A = set_3_model15.get("coefficients")
A.print_rows(num_rows = 2)
# set_4 model:
poly15_data_4 = polynomial_sframe(set_4['sqft_living'], 15)
# my_features = poly15_data_4.column_names() # get the name of the features
poly15_data_4['price'] = set_4['price'] # add price to the data since it's the target
set_4_model15 = graphlab.linear_regression.create(poly15_data_4, 
                                                  target = 'price', 
                                                  features = my_features, 
                                                  l2_penalty= l2_small_penalty,
                                                  validation_set = None,
                                                  verbose = False)
print "############################################"
print "set_4_coefficient:" 
A = set_4_model15.get("coefficients")
A.print_rows(num_rows = 2)

"""
# Ridge regression comes to rescue

Generally, whenever we see weights 
change so much in response to change 
in data, we believe the variance of 
our estimate to be large. 

Ridge regression aims to address 
this issue by penalizing "large" 
weights. (Weights of model15 looked 
quite small, but they are not that 
small because 'sqft_living' input is in the order of thousands.)
With the argument l2_penalty=1e5, 
fit a 15th-order polynomial model on 
set_1, set_2, set_3, and set_4. Other 
than the change in the l2_penalty parameter, 
the code should be the same as the 
experiment above. Also, make sure 
GraphLab Create doesn't create its 
own validation set by using the 
option validation_set = None in this call.
"""
l2_penalty=1e5


# set_1 model:
poly15_data_1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_data_1.column_names() # get the name of the features
poly15_data_1['price'] = set_1['price'] # add price to the data since it's the target
set_1_model15 = graphlab.linear_regression.create(poly15_data_1, 
                                                  target = 'price', 
                                                  features = my_features,
                                                  l2_penalty= l2_penalty,
                                                  validation_set = None,
                                                  verbose = False)
print "############################################"
print "set_1_coefficient for penalty:" 
A = set_1_model15.get("coefficients")
A.print_rows(num_rows = 2)
# set_2 model:
poly15_data_2 = polynomial_sframe(set_2['sqft_living'], 15)
# my_features = poly15_data_2.column_names() # get the name of the features
poly15_data_2['price'] = set_2['price'] # add price to the data since it's the target
set_2_model15 = graphlab.linear_regression.create(poly15_data_2, 
                                                  target = 'price', 
                                                  features = my_features, 
                                                  l2_penalty= l2_penalty,
                                                  validation_set = None,
                                                  verbose = False)
print "############################################"
print "set_2_coefficient for penalty:" 
A = set_2_model15.get("coefficients")
A.print_rows(num_rows = 2)
# set_3 model:
poly15_data_3 = polynomial_sframe(set_3['sqft_living'], 15)
my_features = poly15_data_3.column_names() # get the name of the features
poly15_data_3['price'] = set_3['price'] # add price to the data since it's the target
set_3_model15 = graphlab.linear_regression.create(poly15_data_3, 
                                                  target = 'price', 
                                                  features = my_features, 
                                                  l2_penalty= l2_penalty,
                                                  validation_set = None,
                                                  verbose = False)
print "############################################"
print "set_3_coefficient for penalty:"
A = set_3_model15.get("coefficients")
A.print_rows(num_rows = 2)
# set_4 model:
poly15_data_4 = polynomial_sframe(set_4['sqft_living'], 15)
# my_features = poly15_data_4.column_names() # get the name of the features
poly15_data_4['price'] = set_4['price'] # add price to the data since it's the target
set_4_model15 = graphlab.linear_regression.create(poly15_data_4, 
                                                  target = 'price', 
                                                  features = my_features, 
                                                  l2_penalty= l2_penalty,
                                                  validation_set = None,
                                                  verbose = False)
print "############################################"
print "set_4_coefficient for penalty:" 
A = set_4_model15.get("coefficients")
A.print_rows(num_rows = 2)


"""
## Selecting an L2 penalty via cross-validation

Just like the polynomial degree, 
the L2 penalty is a "magic" parameter 
we need to select. We could use 
the validation set approach as 
we did in the last module, but 
that approach has a major disadvantage: 
it leaves fewer observations available 
for training. Cross-validation seeks 
to overcome this issue by using all 
of the training set in a smart way.

We will implement a kind of cross-validation 
called k-fold cross-validation. 
The method gets its name because 
it involves dividing the training 
set into k segments of roughtly 
equal size. Similar to the validation set 
method, we measure the validation error 
with one of the segments designated as 
the validation set. The major difference 
is that we repeat the process k times as follows:
Set aside segment 0 as the validation set, 
and fit a model on rest of data, and 
evalutate it on this validation set
Set aside segment 1 as the validation set, 
and fit a model on rest of data, and 
evalutate it on this validation set
...
Set aside segment k-1 as the 
validation set, and fit a model 
on rest of data, and evalutate 
it on this validation set
After this process, we compute 
the average of the k validation
errors, and use it as an estimate
of the generalization error. 
Notice that all observations 
are used for both training and 
validation, as we iterate over 
segments of data.
To estimate the generalization 
error well, it is crucial to 
shuffle the training data before 
dividing them into segments. 
GraphLab Create has a utility 
function for shuffling a given 
SFrame. We reserve 10% of the 
data as the test set and shuffle 
the remainder. (Make sure to use 
seed=1 to get consistent answer.)
"""


(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)
"""
Once the data is shuffled, 
we divide it into equal segments. 
Each segment should receive n/k 
elements, where n is the number 
of observations in the training 
set and k is the number of segments. 
Since the segment 0 starts at 
index 0 and contains n/k elements, 
it ends at index (n/k)-1. 

The segment 1 starts where the 
segment 0 left off, at index (n/k). 
With n/k elements, the segment 1 ends 
at index (n*2/k)-1. Continuing in this 
fashion, we deduce that the segment 
i starts at index (n*i/k) and ends 
at (n*(i+1)/k)-1.

With this pattern in mind, we write 
a short loop that prints the starting 
and ending indices of each segment, 
just to make sure you are getting 
the splits right.
"""
n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation
print "###########   Checkind indecies:"
for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)
"""
Let us familiarize ourselves 
with array slicing with SFrame. 
To extract a continuous slice 
from an SFrame, use colon in 
square brackets. For instance, 
the following cell extracts 
rows 0 to 9 of train_valid_shuffled. 
Notice that the first index (0)
is included in the slice but 
the last index (10) is omitted.

train_valid_shuffled[0:10] # rows 0 to 9

Now let us extract individual 
segments with array slicing. 
Consider the scenario where 
we group the houses in the 
train_valid_shuffled dataframe 
into k=10 segments of roughly 
equal size, with starting and 
ending indices computed as above. 
Extract the fourth segment (segment 3) 
and assign it to a variable called 
validation4.
"""
n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation
validation4 = train_valid_shuffled[(n*3)/k : (n*(3+1))/k-1]
"""
To verify that we have the right 
elements extracted, run the following 
cell, which computes the average price 
of the fourth segment. When rounded 
to nearest whole number, the average 
should be $536,234.
"""
print "average of validation4=", int(round(validation4['price'].mean(), 0))

"""
After designating one of the k segments 
as the validation set, we train a 
model using the rest of the data. 
To choose the remainder, we slice (0:start) 
and (end+1:n) of the data and paste 
them together. SFrame has append() 
method that pastes together two 
disjoint sets of rows originating 
from a common dataset. For instance, 
the following cell pastes together 
the first and last two rows of the 
train_valid_shuffled dataframe.
"""
n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
print first_two.append(last_two)

first_part = train_valid_shuffled[0:5818]
second_part = train_valid_shuffled[7758:]
train4 = first_part.append(second_part)
print "average of train4 = ",int(round(train4['price'].mean(), 0))

"""
Now we are ready to implement 
k-fold cross-validation. 
Write a function that computes k validation 
errors by designating each of 
the k segments as the validation 
set. It accepts as parameters 
(i) k, (ii) l2_penalty, (iii) dataframe, (iv) 
name of output column (e.g. price) 
and (v) list of feature names. 
The function returns the average 
validation error using k segments 
as validation sets.

For each i in [0, 1, ..., k-1]:
# Compute starting and ending indices of segment i and call 'start' and 'end'
# Form validation set by taking a slice (start:end+1) from the data.
# Form training set by appending slice (end+1:n) to the end of slice (0:start).
# Train a linear model using training set just formed, with a given l2_penalty
# Compute validation error using validation set just formed
"""



def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    rss_sum = 0
    n = len(data)
    for subset_count in range(1,k+1):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        validation_set = data[start : end+1]
        first_part = data[0 : start]
        second_part = data[end+1:]
        training_set = first_part.append(second_part)
        model = graphlab.linear_regression.create(training_set, 
                                                  target = output_name, 
                                                  features = features_list, 
                                                  l2_penalty=l2_penalty,
                                                  validation_set=None,
                                                  verbose=False)
        predictions = model.predict(validation_set)
        residuals = validation_set['price'] - predictions
        rss = sum(residuals * residuals)
        rss_sum += rss
    validation_error = rss_sum / k # average = sum / size or you can use np.mean(list_of_validation_error)
    return validation_error    
"""
Once we have a function to compute 
the average validation error for 
a model, we can write a loop to 
find the model that minimizes the 
average validation error. Write a 
loop that does the following:
    # We will again be aiming to fit a 15th-order 
      polynomial model using the sqft_living input
    # For l2_penalty in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] 
       (to get this in Python, you can use 
       this Numpy function: np.logspace(1, 7, num=13).)
           % Run 10-fold cross-validation with l2_penalty
    # Report which L2 penalty produced the lowest average validation error.

Note: since the degree of the 
polynomial is now fixed to 15, 
to make things faster, you 
should generate polynomial 
features in advance and re-use 
them throughout the loop. 
Make sure to use train_valid_shuffled 
when generating polynomial features!
"""
poly_data = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
my_features = poly_data.column_names()
poly_data['price'] = train_valid_shuffled['price']

validation_error_dict = {}
for l2_penalty in np.logspace(1, 7, num=13):
    val_err = k_fold_cross_validation(10, l2_penalty, poly_data, 'price', my_features)    
#    print l2_penalty
    validation_error_dict[l2_penalty] = val_err
print validation_error_dict

"""
Once you found the best value 
for the L2 penalty using 
cross-validation, it is important 
to retrain a final model on all 
of the training data using this 
value of l2_penalty. This way, 
your final model will be trained 
on the entire dataset.
"""
best_l2_penalty = 1000
poly_data = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
features_list = poly_data.column_names()
poly_data['price'] = train_valid_shuffled['price']
model = graphlab.linear_regression.create(poly_data, 
                                          target = 'price', 
                                          features = my_features, 
                                          l2_penalty = best_l2_penalty,
                                          validation_set=None,
                                          verbose=False)

poly_test = polynomial_sframe(test['sqft_living'], 15)
predictions = model.predict(poly_test)
errors = predictions-test['price']
rss = (errors*errors).sum()
print "RSS of test data with best L2_Penalty = ", rss



