from __future__ import division
import graphlab
import numpy as np
import math
import string
import json

"""
\*   Implementing logistic regression from scratch   */

The goal of this notebook is 
to implement your own logistic 
regression classifier. You will:
  - Extract features from Amazon 
    product reviews.
  - Convert an SFrame into a 
    NumPy array.
  - Implement the link function 
    for logistic regression.
  - Write a function to compute 
    the derivative of the log 
    likelihood function with 
  - respect to a single coefficient.
    Implement gradient ascent.
  - Given a set of coefficients, predict sentiments.
  - Compute classification accuracy for the logistic regression model.
"""
products = graphlab.SFrame('amazon_baby_subset.gl/')
"""
One column of this dataset is 'sentiment', 
corresponding to the class label with +1 
indicating a review with positive sentiment 
and -1 indicating one with negative sentiment.

Let us quickly explore more of this dataset. 
The 'name' column indicates the name of the 
product. Here we list the first 10 products 
in the dataset. We then count the number of 
positive and negative reviews.
"""

"""
\*   Apply text cleaning on the review data   */

In this section, we will perform some simple 
feature cleaning using SFrames. The last 
assignment used all words in building bag-of-words 
features, but here we limit ourselves to 193 words 
(for simplicity). We compiled a list of 193 most 
frequent words into a JSON file.
Now, we will load these words from this JSON file:
"""
# load these words from this JSON
with open('important_words.json', 'r') as f: # Reads the list of most frequent words
    important_words = json.load(f)
important_words = [str(s) for s in important_words]

"""
Now, we will perform 2 simple 
data transformations:
  - Remove punctuation using Python's built-in string functionality.
  - Compute word counts (only for important_words)
We start with Step 1 which can be done as follows:
"""
# Remove punctuation using Python's built-in string functionality.
def remove_punctuation(text):
    return text.translate(None, string.punctuation) 

products['review_clean'] = products['review'].apply(remove_punctuation)

"""
Now we proceed with Step 2. 
For each word in important_words, 
we compute a count for the number 
of times the word occurs in the 
review. We will store this count 
in a separate column (one for each word). 
The result of this feature processing 
is a single column for each word in 
important_words which keeps a count 
of the number of times the respective 
word occurs in the review text.

Note: There are several ways of 
doing this. In this assignment, 
we use the built-in count function 
for Python lists. Each review string 
is first split into individual words 
and the number of occurances of a 
given word is counted.
"""
# Compute word counts (only for important_words)
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))

# print products['perfect']

"""
Now, write some code to compute 
the number of product reviews that 
contain the word perfect.

Hint: 
   1- First create a column called 
      contains_perfect which is set 
      to 1 if the count of the word 
      perfect (stored in column perfect) 
      is >= 1.
   2- Sum the number of 1s in the column contains_perfect.
"""
products['contains_perfect'] = 0
products['contains_perfect'] = products['perfect'].apply(lambda perfect : +1 if perfect >= 1 else 0)

"""
print "-----------------------------------------------------"
print "Question 1:"
print products['contains_perfect'].sum()
"""

# Convert SFrame to NumPy array
def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)

# convert the data into NumPy arrays.
# Warning: This may take a few minutes!
feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')
"""
print "-----------------------------------------------------"
print "Question 2:"
print "feature_matrix.shape =", feature_matrix.shape
"""
# print feature_matrix

"""
\*   Estimating conditional probability with link function   */

Recall from lecture that the link function is given by:

    P(y_i = +1 | x_i, w) = 1/ (1 + exp(-w^T h(x_i)))

where the feature vector h(x_i) represents 
the word counts of important_words in the 
review x_i. 
Complete the following function that implements the link function:
"""
# produces probablistic estimate for P(y_i = +1 | x_i, w).
# estimate ranges between 0 and 1.
def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients 
    score = - np.dot(feature_matrix, coefficients) 
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1 / (1 + np.exp(score))
    # return predictions
    return predictions

"""
\*   Checkpoint   */
Just to make sure you are on the 
right track, we have provided a 
few examples. If your predict_probability 
function is implemented correctly, 
then the outputs will match:

dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])

correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),          1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_predictions = np.array( [ 1./(1+np.exp(-correct_scores[0])), 1./(1+np.exp(-correct_scores[1])) ] )

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_predictions           =', correct_predictions
print 'output of predict_probability =', predict_probability(dummy_feature_matrix, dummy_coefficients)
"""

"""
\*   Compute derivative of log likelihood with respect to a single coefficient   */
Recall from lecture:

   deriv_j = sum[ h_j(x_i) * (indicator[y_i = +1] - P(y_j=+1|x_i, w)) ]
 
We will now write a function 
that computes the derivative 
of log likelihood with respect 
to a single coefficient  w_j.

The function accepts two arguments:

   * errors vector containing  indicator[y_i = +1] - P(y_i=+1|x_i, w) for all i.
   * feature vector containing h_j(x_i) for all i.
"""
# Compute derivative of log likelihood with respect to a single coefficient
def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.sum(errors * feature)
    # Return the derivative
    return derivative

"""
The log likelihood is computed using 
the following formula 
(see the advanced optional video 
 if you are curious about the 
 derivation of this equation):
   score = w^T h(x_i)
   ll(w) = sum[ (indicator[y_1 = +1] - 1) score - ln(1+exp(-score)) ]
"""
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)
    return lp
# Checkpoint
"""
dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])
dummy_sentiment = np.array([-1, 1])

correct_indicators  = np.array( [ -1==+1,                                       1==+1 ] )
correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),                     1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_first_term  = np.array( [ (correct_indicators[0]-1)*correct_scores[0],  (correct_indicators[1]-1)*correct_scores[1] ] )
correct_second_term = np.array( [ np.log(1. + np.exp(-correct_scores[0])),      np.log(1. + np.exp(-correct_scores[1])) ] )

correct_ll          =      sum( [ correct_first_term[0]-correct_second_term[0], correct_first_term[1]-correct_second_term[1] ] ) 

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_log_likelihood           =', correct_ll
print 'output of compute_log_likelihood =', compute_log_likelihood(dummy_feature_matrix, dummy_sentiment, dummy_coefficients)
"""

"""
\*   Taking gradient steps   */

Now we are ready to implement our 
own logistic regression. All we 
have to do is to write a gradient 
ascent function that takes gradient 
steps towards the optimum.
Complete the following function to 
solve the logistic regression model 
using gradient ascent:
"""
def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):

        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            derivative = feature_derivative(errors, feature_matrix[:,j])
            
            # add the step size times the derivative to the current coefficient
            coefficients[j] = coefficients[j] + (step_size * derivative)
        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients

coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194),
                                   step_size=1e-7, max_iter=301)

"""
\*   Predicting sentiments   */

Recall from lecture that class predictions 
for a data point xx can be computed from 
the coefficients ww using the following formula:
    
    y_hat_i = 1 if 

Now, we will write some code to compute class 
predictions. We will do this in two steps:

    Step 1: First compute the scores using 
            feature_matrix and coefficients 
            using a dot product.

    Step 2: Using the formula above, compute 
            the class predictions from the scores.
"""
scores = np.dot(feature_matrix, coefficients)
boolean_predictions = (scores>0)

"""
print "-----------------------------------------------------"
print "Question 5:"
print np.sum(boolean_predictions)
"""
numerical_predictions = boolean_predictions * 1

num_correct_predictions = np.count_nonzero(numerical_predictions)
num_mistakes = len(numerical_predictions) - num_correct_predictions
accuracy = num_correct_predictions / len(numerical_predictions)
"""
print "-----------------------------------------------------"
print "Question 6:"
print '# Reviews   correctly classified =', len(products) - num_mistakes
print '# Reviews incorrectly classified =', num_mistakes
print '# Reviews total                  =', len(products)
print "-----------------------------------------------------"
print 'Accuracy = %.3f' % accuracy
"""

"""
\*   Which words contribute most to positive & negative sentiments?   */

Recall that in Module 2 assignment, 
we were able to compute the "most positive words". 

These are words that correspond most strongly 
with positive reviews. In order to do this, 
we will first do the following:
Treat each coefficient as a tuple, 
i.e. (word, coefficient_value).
Sort all the (word, coefficient_value) 
tuples by coefficient_value in 
descending order.
"""
coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)
"""
Ten "most positive" words
Now, we compute the 10 words 
that have the most positive 
coefficient values. These words 
are associated with positive sentiment.
"""

"""
print "-----------------------------------------------------"
print "Question 7:"
print "Most Positive Words are"
print ""
print word_coefficient_tuples[0:10]
"""

"""
\*   Ten "most negative" words   */
Next, we repeat this exercise 
on the 10 most negative words. 
That is, we compute the 10 words 
that have the most negative 
coefficient values. These words 
are associated with negative 
sentiment.
"""
"""
print "-----------------------------------------------------"
print "Question 8:"
print "Most Negative Words are"
print ""
print word_coefficient_tuples[-11:]
"""

