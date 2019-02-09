from Node import Node
import operator
import math
import copy

'''Return the most common value of the specified column index'''
def __most_common__(S,index):
    term_set = set()
    term_counts = dict()
    for example in S:
        term_set.add(example[index])
    for term in term_set:
        term_counts[term] = 0
    for term in term_set:
        for example in S:
            if(example[index] == term):
                term_counts[term] += 1
    return max(term_counts,key=term_counts.get)

'''Return the Attribute that best splits the data set'''
def __best_split__(S,Columns,Attributes,Labels,func):
    attr_gains = dict()
    for attribute,a_values in Attributes.items():
        attr_gains[attribute] = __gain__(S,Columns,attribute,a_values,Labels,func)
    return max(attr_gains,key=attr_gains.get)

'''Calculate the information gain of splitting the dataset
S by the given attribute.
func is the function to use to calculate the information gain, can use:
-- __entropy__
-- __gini__
-- __majority_error__
Return a float.'''
def __gain__(S,Columns,attribute,a_values,Labels,func):
    gain = func(S,Labels)
    for a_value in a_values:
        S_v = __subset__(S,Columns,attribute,a_value)
        gain -= (len(S_v)/len(S)*func(S_v,Labels))
    return gain

''' Calculate the entropy of the given dataset with given Labels'''
def __entropy__(S,Labels):
    prob_dict = __probabilities__(S,Labels)
    entropy = 0.0
    for prob in prob_dict.values():
        if prob == 0.0:
            continue
        entropy -= (prob*math.log2(prob))
    return entropy

'''Calculate the Gini Index of the given dataset and given Labels'''
def __gini__(S,Labels):
    prob_dict = __probabilities__(S,Labels)
    gini = 1.0
    for prob in prob_dict.values():
        gini -= (prob**2)
    return gini

'''Calculate the Majority Error for the dataset given Labels'''
def __majority_error__(S,Labels):
    prob_dict = __probabilities__(S,Labels)
    majority_error = 1 - max(prob_dict.values())
    return majority_error

'''Calculate the probability of each Label in S.
Store them in a dictionary {'Label':p} where p is float.'''
def __probabilities__(S,Labels):
    prob_dict = dict()
    for label in Labels:
        prob_dict[label] = 0
    for label in Labels:
        for example in S:
            if example[len(example)-1] == label:
                prob_dict[label] += 1
    for label,count in prob_dict.items():
        if len(S) == 0:
            prob_dict[label] = 0.0
        else:
            prob_dict[label] = count/len(S)
    return prob_dict

'''Returns a subset of S where each example has A=attr_value'''
def __subset__(S,Columns,A,attr_value):
    S_v = []
    for example in S:
        if example[Columns.index(A)] == attr_value:
            S_v.append(example)
    return S_v

'''Perform the ID3 algorithm on dataset S. Return a Node that has an attribute
label with branches to other Nodes, or is a leaf node and contains only a Label.
Column - list of column names in dataset (attribute names)
Attributes - dictionary of attribute name -> values (i.e. Temp:['hot','cold'])
Labels - set of values example can evaluate to (the prediction/result of example)''' 
def ID3(S,Columns,Attributes,Labels,func):
    if(len(Labels) == 1):
        leaf_name = str(Labels.pop())
        #print('Leaf node: ' + leaf_name)
        return Node(leaf_name)
    if(len(Attributes) == 0):
        return Node(str(__most_common__(S,len(Columns)-1)))
    
    A = __best_split__(S,Columns,Attributes,Labels,func)
    #print('Splitting on ' + str(A) + '!')
    root = Node(str(A))
    for attr_value in Attributes[A]:
        S_v = __subset__(S,Columns,A,attr_value)
        if len(S_v) == 0:
            root.branches[attr_value] = Node(str(__most_common__(S,len(Columns)-1)))
        else:
            #print('Attributes: ' + str(len(Attributes)))
            Attributes_v = copy.deepcopy(Attributes)
            Attributes_v.pop(A)
            #print('Attributes_v: ' + str(len(Attributes_v)))
            Labels_v = set()
            for example in S_v:
                Labels_v.add(example[len(example)-1])
            #print('Labels: ' + str(len(Labels)))
            #print('Labels_v: ' + str(len(Labels_v)))
            root.branches[attr_value] = ID3(S_v,Columns,Attributes_v,Labels_v,func)
    return root

def Traverse(node):
    current = node
    
    if current.isLeaf():
        #print('!'+str(current.name)+'!')
        return
    else:
        print(current.name)
        for _,node in current.branches.items():
            #print('-'+str(branch)+'-')
            Traverse(node)

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

S = []

Columns = ['buying','maint','doors','persons','lug_boot','safety','label']

Attributes = {'buying':['vhigh','high','med','low'],
'maint':['vhigh','high','med','low'],
'doors':['2','3','4','5more'],
'persons':['2','4','more'],
'lug_boot':['small','med','big'],
'safety':['low','med','high']}

with open('./car/train.csv', 'r') as train_file:
    for line in train_file:
        terms = line.strip().split(',')
        S.append(terms)

Labels = set()
for example in S:
    Labels.add(example[len(example)-1])

'''prob_dict = __probabilities__(S,Labels)
print(prob_dict)
sum = 0.0
for prob in prob_dict.values():
    sum += prob 
print('Sum of probabilities '+str(sum))
print('Entropy: ' + str(__entropy__(S,Labels)))
print('Best attribute to split on: ' + __best_split__(S,Columns,Attributes,Labels))
'''
tree = ID3(S,Columns,Attributes,Labels,__majority_error__)
success = 0
fail = 0
with open('./car/test.csv','r') as test_file:
    for line in test_file:
        example = line.strip().split(',')
        if Predict(example,tree,Columns):
            success += 1
        else:
            fail += 1

print("Success: " + str(success))
print("Fail: " + str(fail))
e_rate = fail/(success+fail)
print("Error rate: " + str(e_rate))