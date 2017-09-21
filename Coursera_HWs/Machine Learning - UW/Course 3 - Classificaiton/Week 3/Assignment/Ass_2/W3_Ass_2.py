from __future__ import division
import graphlab
import numpy as np
import math
import string
import json

"""
\*  Implementing binary decision trees  */
The goal of this notebook is to implement your own binary decision tree classifier. You will:
Use SFrames to do some feature engineering.
Transform categorical variables into binary variables.
Write a function to compute the number of misclassified examples in an intermediate node.
Write a function to find the best feature to split on.
Build a binary decision tree from scratch.
Make predictions using the decision tree.
Evaluate the accuracy of the decision tree.
Visualize the decision at the root node.
Important Note: In this assignment, we will focus on building decision trees where the data contain only binary (0 or 1) features. This allows us to avoid dealing with:
Multiple intermediate nodes in a split
The thresholding issues of real-valued features.
This assignment may be challenging, so brace yourself :)
"""
loans = graphlab.SFrame('lending-club-data.gl/')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

"""
Unlike the previous assignment where we used several features, in this assignment, we will just be using 4 categorical features:
grade of the loan
the length of the loan term
the home ownership status: own, mortgage, rent
number of years of employment.
Since we are building a binary decision tree, we will have to convert these categorical features to a binary representation in a subsequent section using 1-hot encoding.
"""
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]

"""
Subsample dataset to make sure classes are balanced
Just as we did in the previous 
assignment, we will undersample 
the larger class (safe loans) 
in order to balance out our 
dataset. This means we are 
throwing away many data points. 
We use seed=1 so everyone 
gets the same results.
"""
safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)
"""
Note: There are many approaches 
for dealing with imbalanced 
data, including some where we 
modify the learning algorithm. 
These approaches are beyond the 
scope of this course, but some of 
them are reviewed in "Learning 
from Imbalanced Data" by Haibo 
He and Edwardo A. Garcia, IEEE 
Transactions on Knowledge and 
Data Engineering 21(9) (June 26, 2009) pages 1263-1284
For this assignment, we use the 
simplest possible approach, where 
we subsample the overly represented 
class to get a more balanced dataset. 
In general, and especially when 
the data is highly imbalanced, 
we recommend using more advanced
methods.
"""

"""
Transform categorical data into binary features
In this assignment, we will implement binary decision trees (decision trees for binary features, a specific case of categorical variables taking on two values, e.g., true/false). Since all of our features are currently categorical features, we want to turn them into binary features.
For instance, the home_ownership feature represents the home ownership status of the loanee, which is either own, mortgage or rent. For example, if a data point has the feature
   {'home_ownership': 'RENT'}
we want to turn this into three features:
 { 
   'home_ownership = OWN'      : 0, 
   'home_ownership = MORTGAGE' : 0, 
   'home_ownership = RENT'     : 1
 }
Since this code requires a few Python and GraphLab tricks, feel free to use this block of code as is. Refer to the API documentation for a deeper understanding.
"""
loans_data = risky_loans.append(safe_loans)
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})    
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)
    
    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

features = loans_data.column_names()
features.remove('safe_loans')  # Remove the response variable
features
print "Number of features (after binarizing categorical variables) = %s" % len(features)
loans_data['grade.A']
"""
This column is set to 1 if the loan grade is A and 0 otherwise.
Checkpoint: Make sure the following answers match up.
"""
print "Total number of grade.A loans : %s" % loans_data['grade.A'].sum()
print "Expexted answer               : 6422"

train_data, test_data = loans_data.random_split(.8, seed=1)

"""
Decision tree implementation
In this section, we will implement binary decision trees from scratch. There are several steps involved in building a decision tree. For that reason, we have split the entire assignment into several sections.
Function to count number of mistakes while predicting majority class
Recall from the lecture that prediction at an intermediate node works by predicting the majority class for all data points that belong to this node.
Now, we will write a function that calculates the number of missclassified examples when predicting the majority class. This will be used to help determine which feature is the best to split on at a given node of the tree.
Note: Keep in mind that in order to compute the number of mistakes for a majority classifier, we only need the label (y values) of the data points in the node.
Steps to follow :
Step 1: Calculate the number of safe loans and risky loans.
Step 2: Since we are assuming majority class prediction, all the data points that are not in the majority class are considered mistakes.
Step 3: Return the number of mistakes.
Now, let us write the function intermediate_node_num_mistakes which computes the number of misclassified examples of an intermediate node given the set of labels (y values) of the data points contained in the node. Fill in the places where you find ## YOUR CODE HERE. There are three places in this function for you to fill in.
"""
def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
        
    # Count the number of 1's (safe loans)
    num_of_positive = (labels_in_node == +1).sum()
    # Count the number of -1's (risky loans)
    num_of_negative = (labels_in_node == -1).sum()
    # Return the number of mistakes that the majority classifier makes.
    return np.min((num_of_negative, num_of_positive))
"""
Because there are several steps 
in this assignment, we have 
introduced some stopping points 
where you can check your code 
and make sure it is correct 
before proceeding. To test your 
intermediate_node_num_mistakes 
function, run the following code 
until you get a Test passed!, 
then you should proceed. 
Otherwise, you should spend some 
time figuring out where things went wrong.
"""
# Test case 1
example_labels = graphlab.SArray([-1, -1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 1 failed... try again!'

# Test case 2
example_labels = graphlab.SArray([-1, -1, 1, 1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 2 failed... try again!'
    
# Test case 3
example_labels = graphlab.SArray([-1, -1, -1, -1, -1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 3 failed... try again!'

"""
Function to pick best feature to split on
The function best_splitting_feature takes 3 arguments:
The data (SFrame of data which includes all of the feature columns and label column)
The features to consider for splits (a list of strings of column names to consider for splits)
The name of the target/label column (string)
The function will loop through the list of possible features, and consider splitting on each of them. It will calculate the classification error of each split and return the feature that had the smallest classification error when split on.
Recall that the classification error is defined as follows:
classification error= # mistakes / # total examples
Follow these steps:
Step 1: Loop over each feature in the feature list
Step 2: Within the loop, split the data into two groups: one group where all of the data has feature value 0 or False (we will call this the left split), and one group where all of the data has feature value 1 or True (we will call this the right split). Make sure the left split corresponds with 0 and the right split corresponds with 1 to ensure your implementation fits with our implementation of the tree building process.
Step 3: Calculate the number of misclassified examples in both groups of data and use the above formula to compute the classification error.
Step 4: If the computed error is smaller than the best error found so far, store this feature and its error.
This may seem like a lot, but we have provided pseudocode in the comments in order to help you implement the function correctly.
Note: Remember that since we are only dealing with binary features, we do not have to consider thresholds for real-valued features. This makes the implementation of this function much easier.
Fill in the places where you find ## YOUR CODE HERE. There are five places in this function for you to fill in.
"""
def best_splitting_feature(data, features, target):
    
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        right_split = data[data[feature] == 1]
        
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        left_mistakes = intermediate_node_num_mistakes(left_split[target]) 

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
        
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_feature = feature
            best_error = error
    return best_feature # Return the best feature we found
# test
if best_splitting_feature(train_data, features, 'safe_loans') == 'term. 36 months':
    print 'Test passed!'
else:
    print 'Test failed... try again!'

"""
Building the tree
With the above functions implemented correctly, we are now ready to build our decision tree. Each node in the decision tree is represented as a dictionary which contains the following keys and possible values:

{ 
   'is_leaf'            : True/False.
   'prediction'         : Prediction at the leaf node.
   'left'               : (dictionary corresponding to the left tree).
   'right'              : (dictionary corresponding to the right tree).
   'splitting_feature'  : The feature that this node splits on.
}

First, we will write a function that creates a leaf node given a set of target values. Fill in the places where you find ## YOUR CODE HERE. There are three places in this function for you to fill in.
"""
def create_leaf(target_values):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf':  True  }
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])
    
    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = +1
    else:
        leaf['prediction'] =  -1
    # Return the leaf node
    return leaf

"""
We have provided a function that learns the decision tree recursively and implements 3 stopping conditions:
Stopping condition 1: All data points in a node are from the same class.
Stopping condition 2: No more features to split on.
Additional stopping condition: In addition to the above two stopping conditions covered in lecture, in this assignment we will also consider a stopping condition based on the max_depth of the tree. By not letting the tree grow too deep, we will save computational effort in the learning process.
Now, we will write down the skeleton of the learning algorithm. Fill in the places where you find ## YOUR CODE HERE. There are seven places in this function for you to fill in.
"""
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_num_mistakes(target_values) == 0:  ## YOUR CODE HERE
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == []:   ## YOUR CODE HERE
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:  ## YOUR CODE HERE
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)
    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    ## YOUR CODE HERE
    splitting_feature = best_splitting_feature(data, features, target)

    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1] ## YOUR CODE HERE
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target])

    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth) 

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

# Here is a recursive function to count the nodes in your tree:
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

# Run the following test code to check your implementation. 
# Make sure you get 'Test passed' before proceeding.
small_data_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 3)
if count_nodes(small_data_decision_tree) == 13:
    print 'Test passed!'
else:
    print 'Test failed... try again!'
    print 'Number of nodes found                :', count_nodes(small_data_decision_tree)
    print 'Number of nodes that should be there : 13' 

"""
Build the tree!
Now that all the tests are passing, we will train a tree model on the train_data. Limit the depth to 6 (max_depth = 6) to make sure the algorithm doesn't run for too long. Call this tree my_decision_tree.
Warning: This code block may take 1-2 minutes to learn.
"""
# Make sure to cap the depth at 6 by using max_depth = 6
my_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6)

"""
Making predictions with a decision tree
As discussed in the lecture, we can make predictions from the decision tree with a simple recursive function. Below, we call this function classify, which takes in a learned tree and a test point x to classify. We include an option annotate that describes the prediction path when set to True.
Fill in the places where you find ## YOUR CODE HERE. There is one place in this function for you to fill in.
"""
def classify(tree, x, annotate = False):   
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

print "------------------------------------------------"
test_data[0]
print 'Predicted class: %s ' % classify(my_decision_tree, test_data[0])
print "------------------------------------------------"
print "Question 1, 2, 3"
classify(my_decision_tree, test_data[0], annotate=True)

"""
Evaluating your decision tree
Now, we will write a function to evaluate a decision tree by computing the classification error of the tree on the given dataset.
Again, recall that the classification error is defined as follows:
classification error=# mistakes / # total examples
Now, write a function called evaluate_classification_error that takes in as input:
tree (as described above)
data (an SFrame)
target (a string - the name of the target/label column)
This function should calculate a prediction (class label) for each row in data using the decision tree and return the classification error computed using the above formula. Fill in the places where you find ## YOUR CODE HERE. There is one place in this function for you to fill in.
"""
def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    
    # Once you've made the predictions, calculate the classification error and return it
    num_of_mistakes = (prediction != data[target]).sum()/float(len(data))
    return num_of_mistakes

num_of_mistakes = evaluate_classification_error(my_decision_tree, test_data, target)
print "------------------------------------------------"
print "Question 4"
print num_of_mistakes

"""
Printing out a decision stump
As discussed in the lecture, we can print out a single decision stump (printing out the entire tree is left as an exercise to the curious reader).
"""
def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

print "------------------------------------------------"
print "Question 5"
print_stump(my_decision_tree)

"""
Exploring the intermediate left subtree
The tree is a recursive dictionary, so we do have access to all the nodes! We can use
my_decision_tree['left'] to go left
my_decision_tree['right'] to go right
"""
print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])
# Exploring the left subtree of the left subtree
print "------------------------------------------------"
print "Question 6"
print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature'])
print "------------------------------------------------"
print "Question 7"
print_stump(my_decision_tree['right'], my_decision_tree['splitting_feature'])
