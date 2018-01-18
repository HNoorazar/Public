# W6
from __future__ import division
import graphlab
import numpy as np
import math
import string
import json
import matplotlib.pyplot as plt
graphlab.canvas.set_target('ipynb')
"""
Jupyter NotebookLogout module-9-precision-recall-assignment-blank Last Checkpoint: 3 minutes ago (autosaved) Python 2
Python 2 Kernel errorTrusted
File
Edit
View
Insert
Cell
Kernel
Help

# Exploring precision and recall

The goal of this second notebook is to understand precision-recall in the context of classifiers.

 * Use Amazon review data in its entirety.
 * Train a logistic regression model.
 * Explore various evaluation metrics: accuracy, confusion matrix, precision, recall.
 * Explore how various metrics can be combined to produce a cost of making an error.
 * Explore precision and recall curves.
 
Because we are using the full Amazon review dataset (not a subset of words or reviews), in this assignment we return to using GraphLab Create for its efficiency. As usual, let's start by **firing up GraphLab Create**.

Make sure you have the latest version of GraphLab Create (1.8.3 or later). If you don't find the decision tree module, then you would need to upgrade graphlab-create using

```
   pip install graphlab-create --upgrade
```
See [this page](https://dato.com/download/) for detailed instructions on upgrading.
"""

products = graphlab.SFrame('amazon_baby.gl/')

"""
# Extract word counts and sentiments
As in the first assignment of this course, we compute the word counts for individual words and extract positive and negative sentiments from ratings. To summarize, we perform the following:

1. Remove punctuation.
2. Remove reviews with "neutral" sentiment (rating 3).
3. Set reviews with rating 4 or more to be positive and those with 2 or less to be negative.
"""
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

# Remove punctuation.
review_clean = products['review'].apply(remove_punctuation)

# Count words
products['word_count'] = graphlab.text_analytics.count_words(review_clean)

# Drop neutral sentiment reviews.
products = products[products['rating'] != 3]

# Positive sentiment to +1 and negative sentiment to -1
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

train_data, test_data = products.random_split(.8, seed=1)
"""
## Train a logistic regression classifier

We will now train a logistic 
regression classifier with **sentiment** 
as the target and **word_count** as 
the features. We will set 
`validation_set=None` 
to make sure everyone gets 
exactly the same results.  

Remember, even though we now 
know how to implement logistic 
regression, we will use GraphLab 
Create for its efficiency at 
processing this Amazon dataset 
in its entirety.  The focus of 
this assignment is instead on 
the topic of precision and recall.
"""

model = graphlab.logistic_classifier.create(train_data, target='sentiment',
                                            features=['word_count'],
                                            validation_set=None)

"""
# Model Evaluation
We will explore the advanced model evaluation concepts that were discussed in the lectures.

## Accuracy

One performance metric 
we will use for our more 
advanced exploration is 
accuracy, which we have 
seen many times in past 
assignments.  Recall that 
the accuracy is given by

$$
\mbox{accuracy} = \frac{\mbox{# correctly classified data points}}{\mbox{# total data points}}
$$

To obtain the accuracy 
of our trained models 
using GraphLab Create, 
simply pass the option `metric='accuracy'
to the `evaluate` function. We 
compute the **accuracy** of our 
logistic regression model on the **test_data** as follows:
"""
print "============================================================="
print "Question 1"
accuracy= model.evaluate(test_data, metric='accuracy')['accuracy']
print "Test Accuracy: %s" % accuracy
"""
## Baseline: Majority class prediction

Recall from an earlier assignment that we used the **majority class classifier** as a baseline (i.e reference) model for a point of comparison with a more sophisticated classifier. The majority classifier model predicts the majority class for all data points. 

Typically, a good model should beat the majority class classifier. Since the majority class in this dataset is the positive class (i.e., there are more positive than negative reviews), the accuracy of the majority class classifier can be computed as follows:
"""
baseline = len(test_data[test_data['sentiment'] == 1])/len(test_data)
print "Baseline accuracy (majority class classifier): %s" % baseline

"""
## Confusion Matrix

The accuracy, while convenient, does not tell the whole story. For a fuller picture, we turn to the **confusion matrix**. In the case of binary classification, the confusion matrix is a 2-by-2 matrix laying out correct and incorrect predictions made in each label as follows:
```
              +---------------------------------------------+
              |                Predicted label              |
              +----------------------+----------------------+
              |          (+1)        |         (-1)         |
+-------+-----+----------------------+----------------------+
| True  |(+1) | # of true positives  | # of false negatives |
| label +-----+----------------------+----------------------+
|       |(-1) | # of false positives | # of true negatives  |
+-------+-----+----------------------+----------------------+
```
To print out the confusion matrix for a classifier, use `metric='confusion_matrix'`:
"""

confusion_matrix = model.evaluate(test_data, metric='confusion_matrix')['confusion_matrix']
print "============================================================="
print "Question 2"
print confusion_matrix

"""
## Computing the cost of mistakes


Put yourself in the shoes of a manufacturer that sells a baby product on Amazon.com and you want to monitor your product's reviews in order to respond to complaints.  Even a few negative reviews may generate a lot of bad publicity about the product. So you don't want to miss any reviews with negative sentiments --- you'd rather put up with false alarms about potentially negative reviews instead of missing negative reviews entirely. In other words, **false positives cost more than false negatives**. (It may be the other way around for other scenarios, but let's stick with the manufacturer's scenario for now.)

Suppose you know the costs involved in each kind of mistake: 
1. \$100 for each false positive.
2. \$1 for each false negative.
3. Correctly classified reviews incur no cost.
"""
print "============================================================="
print "Question 3"
print "145,706"

"""
## Precision and Recall

You may not have exact dollar amounts 
for each kind of mistake. Instead, you 
may simply prefer to reduce the percentage 
of false positives to be less than, say, 
3.5% of all positive predictions. This 
is where **precision** comes in:

$$
[\text{precision}] = \frac{[\text{# positive data points with positive predicitions}]}{\text{[# all data points with positive predictions]}} = \frac{[\text{# true positives}]}{[\text{# true positives}] + [\text{# false positives}]}
$$

So to keep the percentage of false positives below 3.5% of positive predictions, we must raise the precision to 96.5% or higher. 

**First**, let us compute the precision of the logistic regression classifier on the **test_data**.
"""
precision = model.evaluate(test_data, metric='precision')['precision']
print "Precision on test data: %s" % precision
print "============================================================="
print "Question 4"
print "1443/(26689 + 1443) = 0.051"

print "============================================================="
print "Question 5"
print "Increase threshold for predicting the positive class (y^=+1)"

"""
A complementary metric is **recall**, which measures the ratio between the number of true positives and that of (ground-truth) positive reviews:

$$
[\text{recall}] = \frac{[\text{# positive data points with positive predicitions}]}{\text{[# all positive data points]}} = \frac{[\text{# true positives}]}{[\text{# true positives}] + [\text{# false negatives}]}
$$

Let us compute the recall on the **test_data**.
"""
print "============================================================="
print "Question 6"
print "26689/(26689+1406) = 0.949"

recall = model.evaluate(test_data, metric='recall')['recall']
print "============================================================="
print "Question 7"
print "Recall on test data: %s" % recall
"""
# Precision-recall tradeoff

In this part, we will explore the trade-off between precision and recall discussed in the lecture.  We first examine what happens when we use a different threshold value for making class predictions.  We then explore a range of threshold values and plot the associated precision-recall curve.  
"""

"""
## Varying the threshold

False positives are costly in our example, so we may want to be more conservative about making positive predictions. To achieve this, instead of thresholding class probabilities at 0.5, we can choose a higher threshold. 

Write a function called `apply_threshold` that accepts two things
* `probabilities` (an SArray of probability values)
* `threshold` (a float between 0 and 1).

The function should return an SArray, where each element is set to +1 or -1 depending whether the corresponding probability exceeds `threshold`.
"""
def apply_threshold(probabilities, threshold):
    ### YOUR CODE GOES HERE
    # +1 if >= threshold and -1 otherwise.
    array = probabilities.apply(lambda x : +1 if x>threshold else -1)
    return array

probabilities = model.predict(test_data, output_type='probability')
predictions_with_default_threshold = apply_threshold(probabilities, 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)
print "============================================================="
print "Question 8"
print "Number of positive predicted reviews (threshold = 0.5): %s" % (predictions_with_default_threshold == 1).sum()
print "Number of positive predicted reviews (threshold = 0.9): %s" % (predictions_with_high_threshold == 1).sum()

"""
## Exploring the associated precision and recall as the threshold varies
By changing the probability threshold, it is possible to influence precision and recall. We can explore this as follows:
"""
# Threshold = 0.5
precision_with_default_threshold = graphlab.evaluation.precision(test_data['sentiment'],
                                        predictions_with_default_threshold)

recall_with_default_threshold = graphlab.evaluation.recall(test_data['sentiment'],
                                        predictions_with_default_threshold)

# Threshold = 0.9
precision_with_high_threshold = graphlab.evaluation.precision(test_data['sentiment'],
                                        predictions_with_high_threshold)
recall_with_high_threshold = graphlab.evaluation.recall(test_data['sentiment'],
                                        predictions_with_high_threshold)
print "============================================================="
print "Question 9"
print "Precision (threshold = 0.5): %s" % precision_with_default_threshold
print "Recall (threshold = 0.5)   : %s" % recall_with_default_threshold
print "Precision (threshold = 0.9): %s" % precision_with_high_threshold
print "Recall (threshold = 0.9)   : %s" % recall_with_high_threshold

"""
## Precision-recall curve

Now, we will explore various different values of tresholds, compute the precision and recall scores, and then plot the precision-recall curve.
"""
threshold_values = np.linspace(0.5, 1, num=100)
print threshold_values
precision_all = []
recall_all = []

probabilities = model.predict(test_data, output_type='probability')
for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    
    precision = graphlab.evaluation.precision(test_data['sentiment'], predictions)
    recall = graphlab.evaluation.recall(test_data['sentiment'], predictions)
    
    precision_all.append(precision)
    recall_all.append(recall)

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})
    # plt.show() I do not know why it take so long when this is here!

plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')

print "============================================================="
print "Question 10"    
print threshold_values[np.array(precision_all) >= 0.965].min()


print "============================================================="
print "Question 11"
threshold = 0.98
predictions = apply_threshold(probabilities, threshold)
confusion_matrix = graphlab.evaluation.confusion_matrix(test_data['sentiment'], predictions)
false_negatives = confusion_matrix[(confusion_matrix['target_label'] == +1) & 
                                   (confusion_matrix['predicted_label'] == -1)]['count'][0]
print "false_negatives count =", false_negatives

"""
Evaluating specific search terms

So far, we looked at the number of false positives for the **entire test set**. In this section, let's select reviews using a specific search term and optimize the precision on these reviews only. After all, a manufacturer would be interested in tuning the false positive rate just for their products (the reviews they want to read) rather than that of the entire set of products on Amazon.

## Precision-Recall on all baby related items

From the **test set**, select all the reviews for all products with the word 'baby' in them.
"""
baby_reviews =  test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]
# Now, let's predict the probability of classifying these reviews as positive:
probabilities = model.predict(baby_reviews, output_type='probability')

# Let's plot the precision-recall curve for the baby_reviews dataset.
# First, let's consider the following threshold_values ranging from 0.5 to 1:
threshold_values = np.linspace(0.5, 1, num=100)
"""
Second, as we did above, let's compute precision and recall for each value in threshold_values on the baby_reviews dataset. Complete the code block below.
"""
precision_all = []
recall_all = []

precision_all = []
recall_all = []
for threshold in threshold_values:
    
    # Make predictions. Use the `apply_threshold` function 
    ## YOUR CODE HERE 
    predictions = apply_threshold(probabilities, threshold)
    
    # Calculate the precision.
    # YOUR CODE HERE
    precision = graphlab.evaluation.precision(baby_reviews['sentiment'], predictions)
    
    # YOUR CODE HERE
    recall = graphlab.evaluation.recall(baby_reviews['sentiment'], predictions)
    
    # Append the precision and recall scores.
    precision_all.append(precision)
    recall_all.append(recall)

print "============================================================="
print "Question 12"
print threshold_values[np.array(precision_all) >= 0.965].min()





