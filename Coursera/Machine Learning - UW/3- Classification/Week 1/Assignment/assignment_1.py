from __future__ import division
import graphlab
import numpy as np
import math
import string

"""
\*    Predicting sentiment from product reviews   */

The goal of this first notebook 
is to explore logistic regression 
and feature engineering with 
existing GraphLab functions.
In this notebook you will use 
product review data from Amazon.com 
to predict whether the sentiments 
about a product (from its reviews) 
are positive or negative.

Use SFrames to do some feature engineering
Train a logistic regression 
model to predict the sentiment 
of product reviews.

Inspect the weights (coefficients) 
of a trained logistic regression model.
Make a prediction (both class and probability) 
of sentiment for a new product review.
Given the logistic regression weights, 
predictors and ground truth labels, 
write a function to compute the accuracy 
of the model.
Inspect the coefficients of the logistic 
regression model and interpret their meanings.
Compare multiple logistic regression models.
Let's get started!
Fire up GraphLab Create
"""

"""
Data preparation
We will use a dataset consisting 
of baby product reviews on Amazon.com.
"""
products = graphlab.SFrame('amazon_baby.gl/')
# print products[269]
"""
\*   Build the word count vector for each review   */
We will perform 2 simple data transformations:
    1- Remove punctuation using Python's built-in string functionality.
    2- Transform the reviews into word-counts.

Aside. In this notebook, we 
remove all punctuations for 
the sake of simplicity. 
A smarter approach to punctuations 
would preserve phrases such 
as "I'd", "would've", "hadn't" and 
so forth. See this page for an 
example of smart handling of 
punctuations.
"""
def remove_punctuation(text):
    return text.translate(None, string.punctuation) 


review_without_punctuation = products['review'].apply(remove_punctuation)
products['word_count'] = graphlab.text_analytics.count_words(review_without_punctuation)
# print "products[269]['word_count'] =", products[269]['word_count']

"""
\*   Extract sentiments   */
We will ignore all reviews with 
rating = 3, since they tend to 
have a neutral sentiment.
"""
# print "len(products) =", len(products)
products = products[products['rating'] != 3]
# print "len(products) =", len(products)

"""
Now, we will assign reviews 
with a rating of 4 or higher 
to be positive reviews, while 
the ones with rating of 2 or 
lower are negative. For the 
sentiment column, we use +1 for 
the positive class label and -1 
for the negative class label.
"""
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
print "Types of products and products['rating'] are ", type(products), type(products['rating'])
"""
Now, we can see that the dataset 
contains an extra column called 
sentiment which is either 
positive (+1) or negative (-1).
"""
"""
\*   Split data into training and test sets   */
Let's perform a train/test split 
with 80% of the data in the training 
set and 20% of the data in the test 
set. 
We use seed=1 so that 
everyone gets the same result.
"""
train_data, test_data = products.random_split(.8, seed=1)
"""
\*   Train a sentiment classifier with logistic regression   */
We will now use logistic regression to 
create a sentiment classifier on the 
training data. This model will use 
the column word_count as a feature 
and the column sentiment as the target. 
We will use validation_set=None to 
obtain same results as everyone else.

Note: This line may take 1-2 minutes.
"""
sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target = 'sentiment',
                                                      features=['word_count'],
                                                      validation_set=None)
# sentiment_model
"""
Aside. You may get a warning to 
the effect of "Terminated due to 
numerical difficulties --- this 
model may not be ideal". It means 
that the quality metric 
(to be covered in Module 3) failed 
to improve in the last iteration of 
the run. The difficulty arises as 
the sentiment model puts too much 
weight on extremely rare words. 
A way to rectify this is to apply 
regularization, to be covered in 
Module 4. Regularization lessens 
the effect of extremely rare words. 
For the purpose of this assignment, 
however, please proceed with the 
model above.
"""
"""
Now that we have fitted 
the model, we can extract 
the weights (coefficients) 
as an SFrame as follows:
"""
weights = sentiment_model.coefficients
# print "weights.column_names() =", weights.column_names()
"""
There are a total of 121713 
coefficients in the model. 
Recall from the lecture that 
positive weights  wjwj  correspond 
to weights that cause positive sentiment, 
while negative weights correspond 
to negative sentiment.
Fill in the following block of 
code to calculate how many weights 
are positive ( >= 0). (Hint: The 
'value' column in SFrame weights 
must be positive ( >= 0)).
"""
num_positive_weights = sum(weights['value'] >= 0 )
num_negative_weights = sum(weights['value'] < 0 )

"""
print "-------------------------------"
print "Question 1:"
print "Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights
"""

"""
\*   Making predictions with logistic regression   */
Now that a model is trained, we can 
make predictions on the test data. 
In this section, we will explore this 
in the context of 3 examples in the 
test dataset. We refer to this set 
of 3 examples as the sample_test_data.
"""

sample_test_data = test_data[10:13]
"""
print "sample_test_data['rating'] =", sample_test_data['rating']
print "sample_test_data ="
print sample_test_data
# Let's dig deeper into the first row of the sample_test_data. Here's the full review:
print ""
print "sample_test_data[0]['review'] =", sample_test_data[0]['review']
print ""
print "sample_test_data[1]['review'] =", sample_test_data[1]['review']
"""
"""
We will now make a class prediction 
for the sample_test_data. 
The sentiment_model should 
predict +1 if the sentiment is 
positive and -1 if the sentiment 
is negative. Recall from the 
lecture that the score 
(sometimes called margin) for the 
logistic regression model is 
defined as:

score_i=w^T h(x_i)
 
where h(x_i) represents 
the features for example  ii . 
We will write some code to obtain 
the scores using GraphLab Create. 
For each row, the score (or margin) 
is a number in the range [-inf, inf].
"""
scores_of_sample_test_data = sentiment_model.predict(sample_test_data, output_type='margin')
# print "type(scores_of_sample_test_data)= ", type(scores_of_sample_test_data)
# print "len =", len(scores_of_sample_test_data)
# print ""
# print "scores =", scores
# print ""

"""
\*   Predicting sentiment   */
These scores can be used to make 
class predictions as follows:
   y_hat = +1  w^T h(x_i)> 0 
   y_hat = -1  w^T h(x_i)<= 0 
 
Using scores, write code to 
calculate y_hat, the class predictions:
"""
y_hat = scores_of_sample_test_data.apply(lambda x : +1 if x > 0 else -1)

"""
print "-------------------------------"
print "y_hat =", y_hat
print "-------------------------------"
print "test"
print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data)
"""

"""
Checkpoint: Make sure your class 
            predictions match with 
            the one obtained from 
            GraphLab Create.

Probability predictions
Recall from the lectures that 
we can also calculate the 
probability predictions from 
the scores using:

P(y_i = +1| x_i, w) = 1 / ( 1 + exp( -w^T h(x_i)) )
Using the variable scores 
calculated previously, write 
code to calculate the probability 
that a sentiment is positive using 
the above formula. For each row, 
the probabilities should be a 
number in the range [0, 1].
"""

def find_probabilities(scores_of_sample_test_data):
    return 1 / (1 + np.exp(-scores_of_sample_test_data)) 
   
probabilities_of_sample_test_data = find_probabilities(scores_of_sample_test_data)
"""
print "-------------------------------"
print "Question 2:"
print "probabilities_of_sample_test_data =", probabilities_of_sample_test_data
"""

# test above code:
# print "Class predictions according to GraphLab Create:" 
# print sentiment_model.predict(sample_test_data, output_type='probability')

"""
\*   Find the most positive (and negative) review   */
We now turn to examining the 
full test dataset, test_data, 
and use GraphLab Create to form
predictions on all of the test
data points for faster
performance.

Using the sentiment_model,
find the 20 reviews in the 
entire test_data with the 
highest probability of being 
classified as a positive 
review. We refer to these as 
the "most positive reviews."
To calculate these top-20 
reviews, use the following steps:

    1- Make probability predictions on 
       test_data using the sentiment_model. 
       (Hint: When you call .predict to
       make predictions on the test data,
       use option output_type='probability'
       to output the probability rather
       than just the most likely class.)
    2- Sort the data according to those
       predictions and pick the top 20.
       (Hint: You can use the .topk method
        on an SFrame to find the top k rows
        sorted according to the value of a
        specified column.)
"""
# The following two lines can be replaced by the third line below. 
# So, why we wrote the function find_probabilities, I do not know!
# test_data_scores = sentiment_model.predict(test_data, output_type='margin')
# probabilities_of_test_data = find_probabilities(test_data_scores) 
test_data_probability = sentiment_model.predict(test_data, output_type='probability')

test_data['test_probability'] = test_data_probability
# top_20_indices = test_data_probability.topk_index(topk=20, reverse=False)
"""
print "-------------------------------"
print "Question 3:"
test_data['name','test_probability'].topk('test_probability', k=20).print_rows(num_rows=20)
"""

"""
Now, let us repeat this exercise 
to find the "most negative reviews." 

Use the prediction probabilities to 
find the 20 reviews in the test_data 
with the lowest probability of being 
classified as a positive review. 
Repeat the same steps above but 
make sure you sort in the opposite 
order.
"""

"""
# We will do this question differently, just for fun!
# However, we could do: 
# test_data['name','test_probability'].topk('test_probability', k=20, reverse=True).print_rows(num_rows=20)
print "-------------------------------"
print "Question 4:"
ascending_sorted_test_data = test_data.sort('test_probability', ascending=True)
two_columns = ascending_sorted_test_data['name','test_probability']
two_columns[0:20].print_rows(num_rows=20)
"""

"""
\*   Compute accuracy of the classifier   */

We will now evaluate the accuracy 
of the trained classifier. Recall 
that the accuracy is given by
     
     accuracy = # correctly classified examples/ # total examples

This can be computed as follows:
Step 1: Use the trained model to 
        compute class predictions 
        (Hint: Use the predict method)

Step 2: Count the number of data 
        points when the predicted 
        class labels match the 
        ground truth labels 
        (called true_labels below).

Step 3: Divide the total number of 
        correct predictions by the 
        total number of data points 
        in the dataset.
Complete the function below to compute the classification accuracy:
"""
def get_classification_accuracy(model, data, true_labels):
    # First get the predictions
    data_scores = model.predict(data, output_type='margin')
    predictions = data_scores.apply(lambda x : +1 if x > 0 else -1)
    # Compute the number of correctly classified examples
    v = predictions - true_labels
    no_correct_predictions = sum(v == 0)
    # Then compute accuracy by dividing num_correct by total number of examples
    accuracy = no_correct_predictions / len(true_labels)
    return accuracy

"""
print "-------------------------------"
print "Question 5:"
print get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
"""

"""
\*   Learn another classifier with fewer words   */
There were a lot of words in the 
model we trained above. We will 
now train a simpler logistic 
regression model using only a 
subset of words that occur in 
the reviews. For this assignment, 
we selected a 20 words to work 
with. These are:
"""

significant_words = ['love', 'great', 'easy', 'old', 
                     'little', 'perfect', 'loves', 
                     'well', 'able', 'car', 'broke', 
                     'less', 'even', 'waste', 'disappointed', 
                     'work', 'product', 'money', 'would', 'return']
"""
For each review, we will use the 
word_count column and trim out all 
words that are not in the 
significant_words list above. 
We will use the SArray dictionary 
trim by keys functionality. 
Note that we are performing 
this on both the training and 
test set.
"""

train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)

"""
\*   Train a logistic regression model on a subset of data   */
We will now build a classifier 
with word_count_subset as the 
feature and sentiment as the target.
"""
simple_model = graphlab.logistic_classifier.create(train_data,
                                                   target = 'sentiment',
                                                   features=['word_count_subset'],
                                                   validation_set=None)
# simple_model
# print get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
# simple_model.coefficients
# simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)
"""
simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)
simple_weights = simple_model.coefficients
print "-------------------------------"
print "Question 7:"
print (sum(simple_weights['value'] > 0) - 1)
"""
v = simple_model.coefficients
"""
print "-------------------------------"
print "Question 8:"
# Question 8
# interesting approach
# Watch and learn
simple_weights = simple_model.coefficients
positive_significant_words = simple_weights[(simple_weights['value'] > 0) & (simple_weights['name'] == "word_count_subset")]['index']
print weights.filter_by(positive_significant_words, 'index')
"""
"""

print "-------------------------------"
print "Question 9:"
print "sentimen model accuracy is ", get_classification_accuracy(sentiment_model, train_data, train_data['sentiment'])
print "simple model accuracy is ", get_classification_accuracy(simple_model, train_data, train_data['sentiment'])
"""

"""
print "-------------------------------"
print "Question 10:"
print "sentimen model accuracy is ", get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
print "simple model accuracy is ", get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
"""

"""
\*   Baseline: Majority class prediction   */

It is quite common to use the majority 
class classifier as the a baseline 
(or reference) model for comparison 
with your classifier model. The majority 
classifier model predicts the majority 
class for all data points. At the very 
least, you should healthily beat the 
majority class classifier, otherwise, 
the model is (usually) pointless.
What is the majority class in the train_data?
"""

num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()

print "-------------------------------"
print "Question 11:"
print "Majority class accuracy is", num_positive /(len(train_data['sentiment']))
