#!/usr/bin/python3
'''
Author: Richard Child
University of Utah CS 5350
Date: March 8, 2019

Script for HW3 Part 2 Problem 2b
The Voted Perceptron algorithm is trained on the
bank-note training data, then tested using the testing data.
This is done for epochs level 10 only.
The average error for epoch 10 is reported.
All the weight vectors and votes for epoch 10 are printed.
'''

import Perceptron

train_D = []
print('Starting HW3 Part 2 Problem 2b!')
print("Reading in the training data")
with open('./bank-note/train.csv','r') as trainFile:
    for line in trainFile:
        example = line.strip().split(',')
        example_float = [float(t) for t in example]
        if example_float[-1] == 0:
            example_float[-1] = -1.0
        train_D.append(example_float)   

print("Reading test data")
test_D = []
with open('./bank-note/test.csv','r') as testFile:
    for line in testFile:
        example = line.strip().split(',')
        example_float = [float(t) for t in example]
        if example_float[-1] == 0:
            example_float[-1] = -1.0
        test_D.append(example_float)
 
print('Voted Perceptron')
print('T\tAvg Error')

# Use Voted Perceptron to create weights vector and votes vector
epochs = [x for x in range(10,11)]
weights = []
votes = []
ITERS = 1
for T in epochs:
    num_errors = 0
    for _ in range(ITERS):
        weights,votes = Perceptron.vote_perceptron(train_D,T,0.1)
        # Calculate average error of weight vector on test data
        for example in test_D:
            actual = example[-1]
            features = example[:-1]
            prediction = Perceptron.__vote_prediction__(weights,votes,features)
            if actual != prediction:
                num_errors += 1
    error_rate = num_errors/(len(test_D)*ITERS)
    print("{0}\t{1}".format(T,error_rate))
    print()
    print("Here are the weights and votes:\n")
    
    for i,weight_v in enumerate(weights):
        print('([',end="")
        for w_element in weight_v:
            print('{0:.2f},'.format(w_element),end="") 
        print('],{0})'.format(votes[i]))

    print('\nTotal number of weight vectors: ' + str(len(weights)))
print()
print('Done with HW3 Part 2 Problem 2b!')
print()