#!/usr/bin/python3
import DecisionTree as DT
import AdaBoost
import statistics

def __adaboost_prediction__(learners,votes,example,Columns,pos_label,neg_label):
    prediction = 0.0
    for i,learner in enumerate(learners):
        learner_prediction = DT.predict(example,learner,Columns)
        if learner_prediction == pos_label:
            prediction += votes[i]
        elif learner_prediction == neg_label:
            prediction -= votes[i]
        else:
            raise Exception("There was a prediction issue")

    if prediction >= 0.0:
        return pos_label
    else:
        return neg_label

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

with open('../DecisionTree/bank/train.csv', 'r') as train_file:
    for line in train_file:
        terms = line.strip().split(',')
        S_train.append(terms)

with open('../DecisionTree/bank/test.csv', 'r') as test_file:
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

print('Iterations\tTrain_E\tTest_E')
with open('./bank_1a_results.txt','w') as fileResults:
    fileResults.write('Iterations\tTrain_E\tTest_E\n')

for T in range(1,1001,100):
    num_train_errors = 0
    num_test_errors = 0
    
    learners,votes = AdaBoost.AdaBoost(T,S_train,Columns,Attributes,Labels)
    print(votes)
    for example in S_train:
        ada_prediction = __adaboost_prediction__(learners,votes,example,Columns,'yes','no')
        if ada_prediction != example[-1]:
            num_train_errors += 1

    for example in S_test:
        ada_prediction = __adaboost_prediction__(learners,votes,example,Columns,'yes','no')
        if ada_prediction != example[-1]:
            num_test_errors += 1

    train_error_rate = num_train_errors/len(S_train)
    test_error_rate = num_test_errors/len(S_test)

    print('{0}\t{1}\t{2}'.format(T,train_error_rate,test_error_rate))
    with open('./bank_1a_results.txt','a') as fileResults:
        fileResults.write('{0}\t{1}\t{2}\n'.format(T,train_error_rate,test_error_rate))