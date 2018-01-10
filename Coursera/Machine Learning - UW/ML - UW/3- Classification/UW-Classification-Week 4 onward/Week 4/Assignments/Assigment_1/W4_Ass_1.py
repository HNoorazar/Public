from __future__ import division
import graphlab
import numpy as np
import math
import string
import json

loans = graphlab.SFrame('lending-club-data.gl/')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

"""
We will be using the same 4 categorical features as in the previous assignment:
grade of the loan
the length of the loan term
the home ownership status: own, mortgage, rent
number of years of employment.
In the dataset, each of these 
features is a categorical feature. 
Since we are building a binary 
decision tree, we will have to 
convert this to binary data in 
a subsequent section using 
1-hot encoding.
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
Just as we did in the 
previous assignment, 
we will undersample 
the larger class (safe loans) 
in order to balance out our 
dataset. This means we are 
throwing away many data 
points. We used seed = 1 
so everyone gets the same results.
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
Note: There are many approaches for 
dealing with imbalanced data, including 
some where we modify the learning algorithm. 
These approaches are beyond the 
scope of this course, but some 
of them are reviewed in this "Learning from Imbalanced Data". 
For this assignment, we use the 
simplest possible approach, where 
we subsample the overly represented 
class to get a more balanced dataset. 
In general, and especially when the 
data is highly imbalanced, we 
recommend using more advanced methods.
"""

"""
Transform categorical data into binary features
Since we are implementing binary decision trees, we transform our categorical data into binary data using 1-hot encoding, just as in the previous assignment. Here is the summary of that discussion:
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
print features

train_data, validation_set = loans_data.random_split(.8, seed=1)

"""
\* Early stopping methods for decision trees */

In this section, we will extend the binary tree implementation from the previous assignment in order to handle some early stopping conditions. Recall the 3 early stopping methods that were discussed in lecture:
Reached a maximum depth. (set by parameter max_depth).
Reached a minimum node size. (set by parameter min_node_size).
Don't split if the gain in error reduction is too small. (set by parameter min_error_reduction).
For the rest of this assignment, we will refer to these three as early stopping conditions 1, 2, and 3.

\* Early stopping condition 1: Maximum depth */

Recall that we already implemented the maximum depth stopping condition in the previous assignment. In this assignment, we will experiment with this condition a bit more and also write code to implement the 2nd and 3rd early stopping conditions.
We will be reusing code from the previous assignment and then building upon this. We will alert you when you reach a function that was part of the previous assignment so that you can simply copy and past your previous code.

\* Early stopping condition 2: Minimum node size */

The function reached_minimum_node_size takes 2 arguments:
The data (from a node)
The minimum number of data points that a node is allowed to split on, min_node_size.
This function simply calculates whether the number of data points at a given node is less than or equal to the specified minimum node size. This function will be used to detect this early stopping condition in the decision_tree_create function.
Fill in the parts of the function below where you find ## YOUR CODE HERE. There is one instance in the function below.
"""
def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    return len(data) <= min_node_size

"""
Early stopping condition 3: Minimum gain in error reduction
The function error_reduction takes 2 arguments:
The error before a split, error_before_split.
The error after a split, error_after_split.
This function computes the gain in error reduction, i.e., the difference between the error before the split and that after the split. This function will be used to detect this early stopping condition in the decision_tree_create function.
Fill in the parts of the function below where you find ## YOUR CODE HERE. There is one instance in the function below.
"""
def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    return (error_before_split - error_after_split)

"""
Grabbing binary decision tree helper functions from past assignment
Recall from the previous assignment that we wrote a function intermediate_node_num_mistakes that calculates the number of misclassified examples when predicting the majority class. This is used to help determine which feature is best to split on at a given node of the tree.
Please copy and paste your code for intermediate_node_num_mistakes here.
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
We then wrote a function best_splitting_feature that finds the best feature to split on given the data and a list of features to consider.
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
        ## YOUR CODE HERE
        right_split =  data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        # YOUR CODE HERE
        left_mistakes = intermediate_node_num_mistakes(left_split[target])            

        # Calculate the number of misclassified examples in the right split.
        ## YOUR CODE HERE
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        ## YOUR CODE HERE
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        ## YOUR CODE HERE
        if error < best_error:
            best_feature = feature            
            best_error = error              
    
    return best_feature # Return the best feature we found

"""
Finally, recall the 
function create_leaf 
from the previous assignment, 
which creates a leaf 
node given a set of 
target values.
"""
def create_leaf(target_values):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf':  True  }   ## YOUR CODE HERE
    
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
\*  Incorporating new early stopping conditions in binary decision tree implementation

Now, you will implement a function that builds a decision tree handling the three early stopping conditions described in this assignment. In particular, you will write code to detect early stopping conditions 2 and 3. You implemented above the functions needed to detect these conditions. The 1st early stopping condition, max_depth, was implemented in the previous assigment and you will not need to reimplement this. In addition to these early stopping conditions, the typical stopping conditions of having no mistakes or no more features to split on (which we denote by "stopping conditions" 1 and 2) are also included as in the previous assignment.

\* Implementing early stopping condition 2: minimum node size:

Step 1: Use the function reached_minimum_node_size that you implemented earlier to write an if condition to detect whether we have hit the base case, i.e., the node does not have enough data points and should be turned into a leaf. Don't forget to use the min_node_size argument.
Step 2: Return a leaf. This line of code should be the same as the other (pre-implemented) stopping conditions.

\* Implementing early stopping condition 3: minimum error reduction:

\* Note: This has to come after finding the best splitting feature so we can calculate the error after splitting in order to calculate the error reduction.
Step 1: Calculate the classification error before splitting. Recall that classification error is defined as:
classification error=# mistakes# total examples
classification error=# mistakes# total examples
Step 2: Calculate the classification error after splitting. This requires calculating the number of mistakes in the left and right splits, and then dividing by the total number of examples.
Step 3: Use the function error_reduction to that you implemented earlier to write an if condition to detect whether the reduction in error is less than the constant provided (min_error_reduction). Don't forget to use that argument.
Step 4: Return a leaf. This line of code should be the same as the other (pre-implemented) stopping conditions.
Fill in the places where you find ## YOUR CODE HERE. There are seven places in this function for you to fill in.
"""
def decision_tree_create(data, features, target, current_depth = 0, 
                         max_depth = 10, min_node_size=1, 
                         min_error_reduction=0.0):
    
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached. All data points have the same target value."                
        return create_leaf(target_values)
    
    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print "Stopping condition 2 reached. No remaining features."                
        return create_leaf(target_values)    
    
    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)
    
    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data, min_node_size):
        print "Early stopping condition 2 reached. Reached minimum node size."
        return create_leaf(target_values)
    
    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples 
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    
    # Calculate the error after splitting (number of misclassified examples 
    # in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:        ## YOUR CODE HERE
        print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values)

    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)        

    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction) 
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}
# Here is a function to count the nodes in your tree:
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

# Run the following test code 
# to check your implementation. 
# Make sure you get 'Test passed' before proceeding.
small_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, 
                                        min_node_size = 10, min_error_reduction=0.0)
if count_nodes(small_decision_tree) == 7:
    print 'Test passed!'
else:
    print 'Test failed... try again!'
    print 'Number of nodes found                :', count_nodes(small_decision_tree)
    print 'Number of nodes that should be there : 7' 

"""
Build a tree!
Now that your code is working, we will train a tree model on the train_data with
max_depth = 6
min_node_size = 100,
min_error_reduction = 0.0
Warning: This code block may take a minute to learn.
"""
my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 100, min_error_reduction=0.0)

"""
Let's now train a tree model ignoring early stopping conditions 2 and 3 so that we get the same tree as in the previous assignment. To ignore these conditions, we set min_node_size=0 and min_error_reduction=-1 (a negative value).
"""
my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)

"""
Making predictions
Recall that in the previous assignment you implemented a function classify to classify a new point x using a given tree.
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

print validation_set[0]
print 'Predicted class: %s ' % classify(my_decision_tree_new, validation_set[0])
print classify(my_decision_tree_new, validation_set[0], annotate = True)
print classify(my_decision_tree_old, validation_set[0], annotate = True)

"""
Evaluating the model
Now let us evaluate the model that we have trained. You implemented this evaluation in the function evaluate_classification_error from the previous assignment.
Please copy and paste your evaluate_classification_error code here.
"""
def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))    
    # Once you've made the predictions, calculate the classification error and return it
    ## YOUR CODE HERE
    num_of_mistakes = (prediction != data[target]).sum()/float(len(data))
    return num_of_mistakes
print "----------------------------------------------"
print "Quiz Question"
print evaluate_classification_error(my_decision_tree_new, validation_set, target)
print evaluate_classification_error(my_decision_tree_old, validation_set, target)

"""
Exploring the effect of max_depth
We will compare three models trained with different values of the stopping criterion. We intentionally picked models at the extreme ends (too small, just right, and too large).
Train three models with these parameters:
model_1: max_depth = 2 (too small)
model_2: max_depth = 6 (just right)
model_3: max_depth = 14 (may be too large)
For each of these three, we set min_node_size = 0 and min_error_reduction = -1.
Note: Each tree can take up to a few minutes to train. In particular, model_3 will probably take the longest to train.
"""
model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, 
                                min_node_size = 0, min_error_reduction=-1)
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 14, 
                                min_node_size = 0, min_error_reduction=-1)
"""
Evaluating the models
Let us evaluate the models on the train and validation data. Let us start by evaluating the classification error on the training data:
"""
print "Training data, class. error (model 1):", evaluate_classification_error(model_1, train_data, target)
print "Training data, class. error (model 2):", evaluate_classification_error(model_2, train_data, target)
print "Training data, class. error (model 3):", evaluate_classification_error(model_3, train_data, target)

"""
Measuring the complexity of the tree
Recall in the lecture that we talked about deeper trees being more complex. We will measure the complexity of the tree as

  complexity(T) = number of leaves in the tree T
Here, we provide a function count_leaves that counts the number of leaves in a tree. Using this implementation, compute the number of nodes in model_1, model_2, and model_3.
"""
def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

print "Number of nodes (model 1):", count_leaves(model_1)
print "Number of nodes (model 2):", count_leaves(model_2)
print "Number of nodes (model 3):", count_leaves(model_3)

"""
Exploring the effect of min_error
We will compare three models trained with different values of the stopping criterion. We intentionally picked models at the extreme ends (negative, just right, and too positive).

Train three models with these parameters:

model_4: min_error_reduction = -1 (ignoring this early stopping condition)
model_5: min_error_reduction = 0 (just right)
model_6: min_error_reduction = 5 (too positive)
For each of these three, we set max_depth = 6, and min_node_size = 0.

Note: Each tree can take up to 30 seconds to train.
"""
model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=0)
model_6 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=5)

print "Validation data, class. error (model 4):", evaluate_classification_error(model_4, validation_set, target)
print "Validation data, class. error (model 5):", evaluate_classification_error(model_5, validation_set, target)
print "Validation data, class. error (model 6):", evaluate_classification_error(model_6, validation_set, target)

print "Number of nodes (model 4):", count_leaves(model_4)
print "Number of nodes (model 5):", count_leaves(model_5)
print "Number of nodes (model 6):", count_leaves(model_6)



    
    
    





