'''
Author: Richard Child

This class implements a Decision Tree learning algorithm, ID3. The
function will return a Node that contains the information about how
to predict labels based on features. The Predict function will follow
the ID3 produced Node and make a prediction.
'''

from Node import Node
import operator
import math
import copy

class DecisionTree:
    def __init__(self, node):
        self.root = node



########################################
####    Several Helper Functions    ####
########################################

'''Return the most common value of the specified column index'''
def __most_common__(S,index):
    term_set = set()
    term_counts = dict()
    for example in S:
        val = example[index]
        term_set.add(val)
    for term in term_set:
        term_counts[term] = 0
    for term in term_set:
        for example in S:
            if(example[index] == term):
                term_counts[term] += 1
    return max(term_counts,key=term_counts.get)

'''Return the Attribute that best splits the data set'''
def __best_split__(S,Columns,Attributes,Labels,func,D):
    attr_gains = dict()
    for attribute,a_values in Attributes.items():
        attr_gains[attribute] = __gain__(S,Columns,attribute,a_values,Labels,func,D)
    return max(attr_gains,key=attr_gains.get)

'''Calculate the information gain of splitting the dataset
S by the given attribute.
func is the function to use to calculate the information gain, can use:
-- __entropy__
-- __gini__
-- __majority_error__
Return a float.'''
def __gain__(S,Columns,attribute,a_values,Labels,func,D):
    gain = func(S,Labels,D)
    for a_value in a_values:
        S_v,D_v = __subset__(S,Columns,attribute,a_value,D)
        gain -= (len(S_v)/len(S)*func(S_v,Labels,D_v))
    return gain

''' Calculate the entropy of the given dataset with given Labels'''
def __entropy__(S,Labels,D):
    prob_dict = __probabilities__(S,Labels,D)
    entropy = 0.0
    for prob in prob_dict.values():
        if prob == 0.0:
            continue
        entropy -= (prob*math.log2(prob))
    return entropy

'''Calculate the Gini Index of the given dataset and given Labels'''
def __gini__(S,Labels,D):
    prob_dict = __probabilities__(S,Labels,D)
    gini = 1.0
    for prob in prob_dict.values():
        gini -= (prob**2)
    return gini

'''Calculate the Majority Error for the dataset given Labels'''
def __majority_error__(S,Labels,D):
    prob_dict = __probabilities__(S,Labels,D)
    majority_error = 1 - max(prob_dict.values())
    return majority_error

'''Calculate the probability of each Label in S.
Store them in a dictionary {'Label':p} where p is float.'''
def __probabilities__(S,Labels,D):
    prob_dict = dict()
    for label in Labels:
        prob_dict[label] = 0
    for label in Labels:
        for i,example in enumerate(S):
            if example[-1] == label:
                prob_dict[label] += D[i]
    for label,count in prob_dict.items():
        if len(S) == 0:
            prob_dict[label] = 0.0
        else:
            prob_dict[label] = count/len(S)
    return prob_dict

'''Returns a subset of S where each example has A=attr_value'''
def __subset__(S,Columns,A,attr_value,D):
    S_v = []
    D_v = []
    for i,example in enumerate(S):
        if example[Columns.index(A)] == attr_value:
            S_v.append(example)
            D_v.append(D[i])
    return S_v,D_v

'''Perform the ID3 algorithm on dataset S. Return a Node that has an attribute
label with branches to other Nodes, or is a leaf node and contains only a Label.
Column - list of column names in dataset (attribute names)
Attributes - dictionary of attribute name -> values (i.e. Temp:['hot','cold'])
Labels - set of values example can evaluate to (the prediction/result of example)''' 
def ID3(S,Columns,Attributes,Labels,D,func=__gini__,max_depth=1,current_depth=0):
    # If all examples have same label, return leaf node
    if(len(Labels) == 1):
        leaf_name = str(Labels.pop())
        return Node(leaf_name)

    # If no more attributes to split on, return leaf with
    # most common label.
    if(len(Attributes) == 0):
        return Node(str(__most_common__(S,len(Columns)-1)))

    # If reached the max tree depth, return leaf node
    # with most common label.
    if max_depth == current_depth:
        return Node(str(__most_common__(S,len(Columns)-1)))

        
    A = __best_split__(S,Columns,Attributes,Labels,func,D)
    root = Node(str(A))
    for attr_value in Attributes[A]:
        S_v,D_v = __subset__(S,Columns,A,attr_value,D)
        if len(S_v) == 0:
            root.branches[attr_value] = Node(str(__most_common__(S,len(Columns)-1)))
        else:
            Attributes_v = copy.deepcopy(Attributes)
            Attributes_v.pop(A)
            Labels_v = set()
            for example in S_v:
                Labels_v.add(example[len(example)-1])
            root.branches[attr_value] = ID3(S_v,Columns,Attributes_v,Labels_v,D_v,func,max_depth,current_depth+1)
    return root

'''Given a test example, use a DecisionTree to predict the test
example's label. If predicted correctly, return True, otherwise 
return False.'''
def Predict(example,tree,Columns):
    actual_label = example[len(example)-1]
    current = tree
    # Go until reach a leaf node
    while not current.isLeaf():
        # Get the attribute to decide on
        decision_attr = current.name 
        # Get the value of that attribute from example
        attr_val = example[Columns.index(decision_attr)]
        # Traverse tree based on example values
        current = current.branches[attr_val] 
    if current.name == actual_label:
        return True
    else:
        return False

'''
Given input example on the tree return the label predicted.
'''
def predict(example,tree,Columns):
    current = tree
    # Go until reach a leaf node
    while not current.isLeaf():
        # Get the attribute to decide on
        decision_attr = current.name 
        # Get the value of that attribute from example
        attr_val = example[Columns.index(decision_attr)]
        # Traverse tree based on example values
        current = current.branches[attr_val] 
    return current.name
