from __future__ import division
import graphlab
import numpy as np
import math
import string
import json
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 6

"""
\* Logistic Regression with L2 regularization */

The goal of this second notebook 
is to implement your own logistic 
regression classifier with L2 
regularization. You will do the 
following:
   * Extract features from Amazon product reviews.
   * Convert an SFrame into a NumPy array.
   * Write a function to compute the derivative 
     of log likelihood function with an L2 penalty 
     with respect to a single coefficient.
   * Implement gradient ascent with an L2 penalty.
   * Empirically explore how the L2 penalty 
     can ameliorate overfitting.
"""

# Load and process review dataset
products = graphlab.SFrame('amazon_baby_subset.gl/')

"""
Just like we did previously, 
we will work with a hand-curated 
list of important words extracted 
from the review data. We will 
also perform 2 simple data 
transformations:
Remove punctuation using Python's built-in string functionality.
Compute word counts (only for the important_words)
Refer to Module 3 assignment for more details.
"""
with open('important_words.json', 'r') as f: # Reads the list of most frequent words
    important_words = json.load(f)
important_words = [str(s) for s in important_words]

def remove_punctuation(text):
    return text.translate(None, string.punctuation) 

# Remove punctuation.
products['review_clean'] = products['review'].apply(remove_punctuation)

# Split out the words into individual columns
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))
"""
print "--------------------------------------------"
print "products \n", 
print products
print "--------------------------------------------"
"""

"""
Train-Validation split
We split the data into a 
train-validation split with 80% 
of the data in the training set 
and 20% of the data in the 
validation set. We use seed=2 
so that everyone gets the same 
result.
Note: In previous assignments, 
we have called this a train-test 
split. However, the portion of 
data that we don't train on will 
be used to help select model 
parameters. Thus, this portion 
of data should be called a 
validation set. Recall that 
examining performance of various 
potential models (i.e. models with 
different parameters) should be 
on a validation set, while 
evaluation of selected model 
should always be on a test set.
"""
train_data, validation_data = products.random_split(.8, seed=2)
# print 'Training set   : %d data points' % len(train_data)
# print 'Validation set : %d data points' % len(validation_data)

def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)

feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment') 

"""
produces probablistic estimate for P(y_i = 1 | x_i, w).
estimate ranges between 0 and 1.
"""
def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients
    scores = np.dot(feature_matrix, coefficients)
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1.0/(1.0 + np.exp(-scores))
    return predictions

"""
\* Adding L2 penalty */

Let us now work on extending 
logistic regression with L2 
regularization. As discussed in 
the lectures, the L2 regularization 
is particularly useful in preventing 
overfitting. In this assignment, we 
will explore L2 regularization in detail.

Recall from lecture and the 
previous assignment that for 
logistic regression without 
an L2 penalty, the derivative 
of the log likelihood function is: 
         
         partial l / partial w_j = sum h_j(x_i) (1[y_i=1] - P(y_i=1| x_i,w))

\* Adding L2 penalty to the derivative */

It takes only a small modification 
to add a L2 penalty. All terms 
indicated in red refer to terms 
that were added due to an L2 penalty.
Recall from the lecture that the 
link function is still the sigmoid:

       P(y_i=1|x_i, w) = 1 / 1 + exp(-w^t h(x_i)) 

We add the L2 penalty term to the per-coefficient derivative of log likelihood:
      
      partial l / partial w_j = sum h_j(x_i) (1[y_i=1] - P(y_i=1| x_i,w)) - 2 lambda w_j
      
The per-coefficient derivative for logistic regression with an L2 penalty is as follows:

       partial l / partial w_j = sum h_j(x_i) (1[y_i=1] - P(y_i=1| x_i,w)) - 2 lambda w_j

and for the intercept term, we have

       partial l / partial w_0 = sum h_0(x_i) (1[y_i=1] - P(y_i=1| x_i,w))

Note: As we did in the Regression 
      course, we do not apply the L2 
      penalty on the intercept. A 
      large intercept does not necessarily 
      indicate overfitting because 
      the intercept is not associated 
      with any particular feature.

Write a function that computes 
the derivative of log likelihood 
with respect to a single coefficient 
w_j. Unlike its counterpart in 
the last assignment, the function 
accepts five arguments:
errors vector containing 
     (1[y_i=1])-P(y_i=1 | x_i,w) for all i.
feature vector containing h_j(x_i) for all i
coefficient containing the 
current value of coefficient wj.
l2_penalty representing the 
L2 penalty constant lambda
feature_is_constant telling 
whether the j-th feature is constant 
or not.

"""

def feature_derivative_with_L2(errors, feature, coefficient, l2_penalty, feature_is_constant):
    
    # Compute the dot product of errors and feature
    derivative = np.sum(np.dot(errors, feature))

    # add L2 penalty term for any feature that isn't the intercept.
    if not feature_is_constant:
        derivative = derivative - 2 * l2_penalty * coefficient
    return derivative

def compute_log_likelihood_with_L2(feature_matrix, sentiment, 
                                   coefficients, l2_penalty):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)    
    lp = np.sum((indicator-1) * scores - np.log(1. + np.exp(-scores))) - l2_penalty*np.sum(coefficients[1:]**2)
    return lp

def logistic_regression_with_L2(feature_matrix, sentiment, 
                                initial_coefficients, 
                                step_size, l2_penalty, 
                                max_iter):
    
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            is_intercept = (j == 0)
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            derivative = feature_derivative_with_L2(errors, feature_matrix[:,j], 
                                                    coefficients[j], l2_penalty, 
                                                    is_intercept)
            # add the step size times the derivative to the current coefficient
            coefficients[j] += step_size * derivative 
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients

# run with L2 = 0
coefficients_0_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                     initial_coefficients=np.zeros(194),
                                                     step_size=5e-6, l2_penalty=0, max_iter=501)
# run with L2 = 4
coefficients_4_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                      initial_coefficients=np.zeros(194),
                                                      step_size=5e-6, l2_penalty=4, max_iter=501)
# run with L2 = 10
coefficients_10_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                      initial_coefficients=np.zeros(194),
                                                      step_size=5e-6, l2_penalty=10, max_iter=501)
# run with L2 = 1e2
coefficients_1e2_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                       initial_coefficients=np.zeros(194),
                                                       step_size=5e-6, l2_penalty=1e2, max_iter=501)
# run with L2 = 1e3
coefficients_1e3_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                       initial_coefficients=np.zeros(194),
                                                       step_size=5e-6, l2_penalty=1e3, max_iter=501)
# run with L2 = 1e5
coefficients_1e5_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                       initial_coefficients=np.zeros(194),
                                                       step_size=5e-6, l2_penalty=1e5, max_iter=501)

"""
\* Compare coefficients */

We now compare the coefficients 
for each of the models that were 
trained above. We will create a 
table of features and learned 
coefficients associated with each 
of the different L2 penalty values.
Below is a simple helper function 
that will help us create this table.
"""
table = graphlab.SFrame({'word': ['(intercept)'] + important_words})
def add_coefficients_to_table(coefficients, column_name):
    table[column_name] = coefficients
    return table

add_coefficients_to_table(coefficients_0_penalty, 'coefficients [L2=0]')
add_coefficients_to_table(coefficients_4_penalty, 'coefficients [L2=4]')
add_coefficients_to_table(coefficients_10_penalty, 'coefficients [L2=10]')
add_coefficients_to_table(coefficients_1e2_penalty, 'coefficients [L2=1e2]')
add_coefficients_to_table(coefficients_1e3_penalty, 'coefficients [L2=1e3]')
add_coefficients_to_table(coefficients_1e5_penalty, 'coefficients [L2=1e5]')

"""
Using the coefficients trained 
with L2 penalty 0, find the 5 most 
positive words (with largest 
positive coefficients). Save them 
to positive_words. Similarly, find 
the 5 most negative words (with 
largest negative coefficients) 
and save them to negative_words.
"""
table[['word','coefficients [L2=0]']].sort('coefficients [L2=0]', ascending = False)[0:5]
negative_words = table.sort('coefficients [L2=0]', ascending = True)[0:5]['word']
positive_words = table.sort('coefficients [L2=0]', ascending = False)[0:5]['word']
"""
print "-----------------------------------------------------"
print "Question 3"
print positive_words
print negative_words
"""

def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    
    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')
    
    table_positive_words = table.filter_by(column_name='word', values=positive_words)
    table_negative_words = table.filter_by(column_name='word', values=negative_words)
    del table_positive_words['word']
    del table_negative_words['word']
    
    for i in xrange(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words[i:i+1].to_numpy().flatten(),
                 '-', label=positive_words[i], linewidth=4.0, color=color)
        
    for i in xrange(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words[i:i+1].to_numpy().flatten(),
                 '-', label=negative_words[i], linewidth=4.0, color=color)
        
    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()
make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list=[0, 4, 10, 1e2, 1e3, 1e5])

"""
\* Measuring accuracy */

Now, let us compute the accuracy 
of the classifier model. Recall 
that the accuracy is given by

       accuracy = # correctly classified data points / # total data points

Recall from lecture that that the 
class prediction is calculated using

y_hat_i = +1 of score is positive and -1 if score is non-positive.

Note: It is important to know that 
the model prediction code doesn't 
change even with the addition of 
an L2 penalty. The only thing that 
changes is the estimated coefficients 
used in this prediction.
Based on the above, we will use the 
same code that was used in Module 3 
assignment.
"""
def get_classification_accuracy(feature_matrix, sentiment, coefficients):
    scores = np.dot(feature_matrix, coefficients)
    apply_threshold = np.vectorize(lambda x: 1. if x > 0  else -1.)
    predictions = apply_threshold(scores)
    
    num_correct = (predictions == sentiment).sum()
    accuracy = num_correct / len(feature_matrix)    
    return accuracy

train_accuracy = {}
train_accuracy[0]   = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_0_penalty)
train_accuracy[4]   = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_4_penalty)
train_accuracy[10]  = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_10_penalty)
train_accuracy[1e2] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e2_penalty)
train_accuracy[1e3] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e3_penalty)
train_accuracy[1e5] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e5_penalty)

validation_accuracy = {}
validation_accuracy[0]   = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_0_penalty)
validation_accuracy[4]   = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_4_penalty)
validation_accuracy[10]  = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_10_penalty)
validation_accuracy[1e2] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e2_penalty)
validation_accuracy[1e3] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e3_penalty)
validation_accuracy[1e5] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e5_penalty)

# Build a simple report
for key in sorted(validation_accuracy.keys()):
    print "L2 penalty = %g" % key
    print "train accuracy = %s, validation_accuracy = %s" % (train_accuracy[key], validation_accuracy[key])
    print "--------------------------------------------------------------------------------"

