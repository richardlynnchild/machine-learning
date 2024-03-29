#!/usr/bin/python3
'''
Author: Richard Child

This file is used to test Decision Tree implementation and the handling
of unknown values in the data set. A sample data set is loaded, numeric
features are converted to binary ones, and unknown values are replaced
using a most-common value heuristic. Several decision trees are created
then used to predict outcomes on the testing data set. The max tree depth
is varied and the results are written to a .csv file in the current directory.

Used to answer HW1, Section 2, Question 3b.
'''

import DecisionTree as DT
import statistics

'''Return the most common value of the specified column index'''
def __most_common_ignore_unknowns__(S,index):
    term_set = set()
    term_counts = dict()
    for example in S:
        val = example[index]
        if val == 'unknown':
            continue
        else:
            term_set.add(val)
    for term in term_set:
        term_counts[term] = 0
    for term in term_set:
        for example in S:
            if(example[index] == term):
                term_counts[term] += 1
    return max(term_counts,key=term_counts.get)

S_train = []
S_test = []

Columns = ['age','job','marital','education','default','balance',
            'housing','loan','contact','day','month','duration',
            'campaign','pdays','previous','poutcome','y']

Attributes = {
    'age':['high','low'],
    'job':['admin.','unknown','unemployed','management','housemaid',
        'entrepreneur','student','blue-collar','self-employed','retired',
        'technician','services'],
    'marital':['married','divorced','single'],
    'education':['unknown','secondary','primary','tertiary'],
    'default':['yes','no'],
    'balance':['high','low'],
    'housing':['yes','no'],
    'loan':['yes','no'],
    'contact':['unknown','telephone','cellular'],
    'day':['high','low'],
    'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
    'duration':['high','low'],
    'campaign':['high','low'],
    'pdays':['high','low'],
    'previous':['high','low'],
    'poutcome':['unknown','other','failure','success']
}

Labels = {'yes','no'}

with open('./bank/train.csv', 'r') as train_file:
    for line in train_file:
        terms = line.strip().split(',')
        S_train.append(terms)

with open('./bank/test.csv', 'r') as test_file:
    for line in test_file:
        terms = line.strip().split(',')
        S_test.append(terms)
# Calculate the median values for each of the numeric attributes
medians = {'age':0.0,'balance':0.0,'day':0.0,'duration':0.0,'campaign':0.0,
            'pdays':0.0,'previous':0.0}

for attr in medians.keys():
    S_attr = []
    for example in S_train:
        S_attr.append(float(example[Columns.index(attr)]))
    medians[attr] = statistics.median(S_attr)

# Now change the numberic values in the dataset to 'high' or 'low' depending
# on if they are above or below the median value for that attribute.
for attr,median in medians.items():
    for example in S_train:
        S_attr_val = float(example[Columns.index(attr)])
        if S_attr_val < median:
            example[Columns.index(attr)] = 'low'
        else:
            example[Columns.index(attr)] = 'high'

    for example in S_test:
        S_attr_val = float(example[Columns.index(attr)])
        if S_attr_val < median:
            example[Columns.index(attr)] = 'low'
        else:
            example[Columns.index(attr)] = 'high'

# Now we eliminate 'unknown' attribute values by substituting them with the
# most common value for that attribute.

most_common_vals_train = dict()
most_common_vals_test = dict()
for attribute in Columns:
    most_common_vals_train[attribute] = __most_common_ignore_unknowns__(S_train,Columns.index(attribute))
    most_common_vals_test[attribute] = __most_common_ignore_unknowns__(S_test,Columns.index(attribute))

for example in S_train:
    for index in range(0,len(Columns)):
        if example[index] == 'unknown':
            example[index] = most_common_vals_train[Columns[index]]

for example in S_test:
    for index in range(0,len(Columns)):
        if example[index] == 'unknown':
            example[index] = most_common_vals_test[Columns[index]] 

with open('./bank3b_results.csv','w') as results_file:
    results_file.write('Depth,Train_ME,Train_Gini,Train_Entropy,Test_ME,' +
                        'Test_Gini,Test_Entropy\n')

print('Running ID3 implementation for Question 3b')
print('Generating decision trees...')
print('Calculating prediction error rates...')
for max_depth in range(1,17):
    me = DT.ID3(S_train,Columns,Attributes,Labels,DT.__majority_error__,max_depth,0)
    gini = DT.ID3(S_train,Columns,Attributes,Labels,DT.__gini__,max_depth,0)
    entropy = DT.ID3(S_train,Columns,Attributes,Labels,DT.__entropy__,max_depth,0)

    train_me_success = 0
    train_gini_success = 0
    train_entropy_success = 0
    train_total = 0

    test_me_success = 0
    test_gini_success = 0
    test_entropy_success = 0
    test_total = 0

    for example in S_train:
        if DT.Predict(example,me,Columns):
            train_me_success += 1
        if DT.Predict(example,gini,Columns):
            train_gini_success += 1
        if DT.Predict(example,entropy,Columns):
            train_entropy_success += 1
        train_total += 1

    for example in S_test:
        if DT.Predict(example,me,Columns):
            test_me_success += 1
        if DT.Predict(example,gini,Columns):
            test_gini_success += 1
        if DT.Predict(example,entropy,Columns):
            test_entropy_success += 1
        test_total += 1

    train_me_er = 1-(train_me_success/train_total)
    train_gini_er = 1-(train_gini_success/train_total)
    train_entropy_er = 1-(train_entropy_success/train_total)

    test_me_er = 1-(test_me_success/test_total)
    test_gini_er = 1-(test_gini_success/test_total)
    test_entropy_er = 1-(test_entropy_success/test_total)

    with open('./bank3b_results.csv','a') as results_file:
        results_file.write('{0},{1:0.3f},{2:0.3f},{3:0.3f},{4:0.3f},{5:0.3f},{6:0.3f}\n'.format(
            max_depth,train_me_er,train_gini_er,train_entropy_er,
            test_me_er,test_gini_er,test_entropy_er))

print('Results written to bank3b_results.csv')
print('Finished with Question 3b!')