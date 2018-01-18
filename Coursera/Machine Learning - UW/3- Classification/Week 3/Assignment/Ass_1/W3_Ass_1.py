from __future__ import division
import graphlab
import numpy as np
import math
import string
import json
graphlab.canvas.set_target('ipynb')

"""
\*  Identifying safe loans with decision trees  */

The LendingClub is a peer-to-peer 
leading company that directly 
connects borrowers and potential 
lenders/investors. In this 
notebook, you will build a 
classification model to predict 
whether or not a loan provided 
by LendingClub is likely to default.

In this notebook you will use 
data from the LendingClub to 
predict whether a loan will 
be paid off in full or the loan 
will be charged off and possibly 
go into default. In this assignment 
you will:
   * Use SFrames to do some feature engineering.
   * Train a decision-tree on the LendingClub dataset.
   * Visualize the tree.
   * Predict whether a loan will default 
     along with prediction probabilities 
     (on a validation set).
   * Train a complex tree model and 
     compare it to simple tree model.
"""

loans = graphlab.SFrame('lending-club-data.gl/')
loans.column_names()
"""
We can see that over half of the 
loan grades are assigned values 
B or C. Each loan is assigned one 
of these grades, along with a more 
finely discretized feature called 
sub_grade (feel free to explore 
that feature column as well!). 
These values depend on the loan 
application and credit report, 
and determine the interest rate 
of the loan. More information 
can be found here.

https://www.lendingclub.com/public/rates-and-fees.action
"""
loans['grade'].show()
loans['home_ownership'].show()

"""
\*  Exploring the target column  */
The target column (label column) 
of the dataset that we are 
interested in is called bad_loans. 
In this column 1 means a risky 
(bad) loan 0 means a safe loan.
In order to make this more intuitive 
and consistent with the lectures, we 
reassign the target to be:
    +1 as a safe loan,
    -1 as a risky (bad) loan.
We put this in a new column called 
safe_loans.
"""
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')
loans['safe_loans'].show(view = 'Categorical')
# You should have:
# Around 81% safe loans
# Around 19% risky loans


"""
\*  Features for the classification algorithm  */
In this assignment, we will be 
using a subset of features 
(categorical and numeric). 
The features we will be using 
are described in the code comments 
below. If you are a finance geek, 
the LendingClub website has a lot 
more details about these features.
"""
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                   # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

"""
\*  Sample data to balance classes  */
As we explored above, our data is 
disproportionally full of safe loans. 
Let's create two datasets: one with 
just the safe loans (safe_loans_raw) 
and one with just the risky loans (risky_loans_raw).
"""
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)
total_no_loans = (len(safe_loans_raw) + len(risky_loans_raw) )
print "Percentage of safe loans  :", len(safe_loans_raw) / total_no_loans
print "Percentage of risky loans :", len(risky_loans_raw) / total_no_loans

"""
One way to combat class imbalance 
is to undersample the larger class 
until the class distribution is 
approximately half and half. Here, 
we will undersample the larger class 
(safe loans) in order to balance out 
our dataset. This means we are throwing 
away many data points. We used seed=1 
so everyone gets the same results.
"""

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)
# Now, let's verify that the resulting 
# percentage of safe and risky loans 
# are each nearly 50%.
print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

"""
Note: There are many approaches 
for dealing with imbalanced data, 
including some where we modify the 
learning algorithm. These approaches 
are beyond the scope of this course, 
but some of them are reviewed in 
this paper. For this assignment, 
we use the simplest possible approach, 
where we subsample the overly represented 
class to get a more balanced dataset. 
In general, and especially when the 
data is highly imbalanced, we 
recommend using more advanced methods.

http://ieeexplore.ieee.org/document/5128907/?reload=true&tp=&arnumber=5128907&url=http:%2F%2Fieeexplore.ieee.org%2Fiel5%2F69%2F5173046%2F05128907.pdf%3Farnumber%3D5128907
"""
train_data, validation_data = loans_data.random_split(.8, seed=1)

"""
Use decision tree to build a classifier
Now, let's use the built-in 
GraphLab Create decision tree 
learner to create a loan 
prediction model on the training 
data. (In the next assignment, 
you will implement your own 
decision tree learning algorithm.) 
Our feature columns and target 
column have already been decided 
above. Use validation_set=None 
to get the same results as everyone else.
"""
decision_tree_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                                target = target, features = features)
decision_tree_model.show(view="Tree")

"""
Visualizing a learned model
As noted in the documentation, 
typically the the max depth of 
the tree is capped at 6. However, 
such a tree can be hard to visualize 
graphically. Here, we instead learn 
a smaller model with max depth of 
2 to gain some intuition by 
visualizing the learned tree.
"""
small_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                   target = target, features = features, max_depth = 2)
"""
Note: To better understand this visual:

The root node is represented using pink.
Intermediate nodes are in green.
Leaf nodes in blue and orange.
"""
small_model.show(view="Tree")


"""
\*  Making predictions  */
Let's consider two positive 
and two negative examples from 
the validation set and see what 
the model predicts. We will do the following:
   Predict whether or not a loan is safe.
   Predict the probability that a loan is safe.
"""
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

"""
\*  Explore label predictions  */
Now, we will use our model 
to predict whether or not 
a loan is likely to default. 
For each row in the sample_validation_data, 
use the decision_tree_model 
to predict whether or not 
the loan is classified as 
a safe loan.
Hint: Be sure to use the .predict() method.
"""
print "--------------------------------------------------"
print "Question 1"
print decision_tree_model.predict(sample_validation_data)
print sample_validation_data['safe_loans']

"""
\*   Explore probability predictions   */
For each row in the sample_validation_data, 
what is the probability 
(according decision_tree_model) 
of a loan being classified as safe?
Hint: Set output_type='probability' 
to make probability predictions 
using decision_tree_model on sample_validation_data:
"""
print "--------------------------------------------------"
print "Question 2"
print [round(p, 2) for p in decision_tree_model.predict(sample_validation_data, output_type='probability')]

"""
Checkpoint: Can you verify that 
for all the predictions with 
probability >= 0.5, the model 
predicted the label +1?
Tricky predictions!
Now, we will explore something 
pretty interesting. For each 
row in the sample_validation_data, 
what is the probability 
(according to small_model) of 
a loan being classified as safe?
Hint: Set output_type='probability' 
to make probability predictions 
using small_model on 
sample_validation_data:
"""
print "--------------------------------------------------"
print "Question 3"
print [round(p, 2) for p in small_model.predict(sample_validation_data, output_type='probability')]

"""
Visualize the prediction on a tree
Note that you should be 
able to look at the small 
tree, traverse it yourself, 
and visualize the prediction 
being made. Consider the 
following point in the 
sample_validation_data
"""
sample_validation_data[1]
small_model.show(view="Tree")
"""
Note: In the tree visualization 
above, the values at the leaf
nodes are not class predictions 
but scores (a slightly advanced 
concept that is out of the scope 
of this course). You can read 
more about this here. If the 
score is $\geq$ 0, the class 
+1 is predicted. Otherwise, 
if the score < 0, we predict class -1.
"""
print "--------------------------------------------------"
print "Question 4"
print small_model.predict(sample_validation_data[1])

"""
\*  Evaluating accuracy of the decision tree model  */

Recall that the accuracy is defined as follows:

    accuracy=# correctly classified examples / # total examples
    
Let us start by evaluating the 
accuracy of the small_model and 
decision_tree_model on the training data
"""
print small_model.evaluate(train_data)['accuracy']
print decision_tree_model.evaluate(train_data)['accuracy']

"""
Checkpoint: You should see 
      that the small_model performs 
      worse than the decision_tree_model 
      on the training data.
Now, let us evaluate the accuracy 
of the small_model and 
decision_tree_model on the entire
validation_data, not just the 
subsample considered above.
"""
print "--------------------------------------------------"
print "Question 5"
print small_model.evaluate(validation_data)['accuracy']
print "decision tree accuracy on validation is:", round(decision_tree_model.evaluate(validation_data)['accuracy'], 2)

"""
Evaluating accuracy of a complex decision tree model
Here, we will train a large 
decision tree with max_depth=10. 
This will allow the learned tree 
to become very deep, and result 
in a very complex model. Recall 
that in lecture, we prefer simpler 
models with similar predictive power. 
This will be an example of a more 
complicated model which has similar 
predictive power, i.e. something we don't want.
"""
big_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                   target = target, features = features, max_depth = 10)
print big_model.evaluate(train_data)['accuracy']
print "big model accuracy on validation is:", big_model.evaluate(validation_data)['accuracy']

"""
Quantifying the cost of mistakes
Every mistake the model makes costs money. In this section, we will try and quantify the cost of each mistake made by the model.
Assume the following:
False negatives: Loans that were actually safe but were predicted to be risky. This results in an oppurtunity cost of losing a loan that would have otherwise been accepted.
False positives: Loans that were actually risky but were predicted to be safe. These are much more expensive because it results in a risky loan being given.
Correct predictions: All correct predictions don't typically incur any cost.
Let's write code that can compute the cost of mistakes made by the model. Complete the following 4 steps:
First, let us compute the predictions made by the model.
Second, compute the number of false positives.
Third, compute the number of false negatives.
Finally, compute the cost of mistakes made by the model by adding up the costs of true positives and false positives.
First, let us make predictions on validation_data using the decision_tree_model:
"""
predictions = decision_tree_model.predict(validation_data)
true_label = validation_data['safe_loans']
"""
False positives are predictions 
where the model predicts +1 but 
the true label is -1. Complete 
the following code block for 
the number of false positives:
"""
false_positive = sum((predictions == +1) & (true_label == -1))
print "false positive", false_positive

false_negative = sum((predictions == -1) & (true_label == +1))
print "false negative", false_negative
print "--------------------------------------------------"
print "Question 6"
print "our_cost =", false_negative*10000 + false_positive*20000



