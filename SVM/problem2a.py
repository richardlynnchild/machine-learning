'''
Author: Richard Child
University of Utah CS 5350
Date: April 5, 2019

Script for HW4 Part 2 Problem 2a
'''

import SVM
import matplotlib.pyplot as plt  

print('Starting HW3 Part 2 Problem 2a!')
train_D = []
print("Reading in the training data")
with open('./bank-note/train.csv','r') as trainFile:
    for line in trainFile:
        example = line.strip().split(',')
        example_float = [float(t) for t in example]
        if example_float[-1] == 0:
            example_float[-1] = -1.0

        # fold in the bias term    
        example_float.insert(len(example_float)-1,1.0)
        train_D.append(example_float)   

print("Reading test data")
test_D = []
with open('./bank-note/test.csv','r') as testFile:
    for line in testFile:
        example = line.strip().split(',')
        example_float = [float(t) for t in example]
        if example_float[-1] == 0:
            example_float[-1] = -1.0

        # fold in the bias term    
        example_float.insert(len(example_float)-1,1.0)        
        test_D.append(example_float)

C = 500/873
w_train = SVM.svm_primal(train_D,100,C,0.1,0)
print(w_train)