#!/usr/bin/python3
'''
Author: Richard Child
University of Utah CS 5350
Date: March 8, 2019

Script for HW3 Part 2 Problem 2a
The Standard Perceptron algorithm is trained on the
bank-note training data, then tested using the testing data.
This is done for epochs ranging from 1 to 10.
The average error for each epoch level is reported.
The learned weight vector for T=10 is reported.
The average error rates are plotted in figure_2a.pdf.
'''

import Perceptron
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
 
print('T\tAvg Error')
# Use Standard Perceptron to create weight vector
avg_error_rates = []
epochs = [x for x in range(1,11)]
weights = []
ITERS = 50
for T in epochs:
    num_errors = 0
    for _ in range(ITERS):
        weights = Perceptron.perceptron(train_D,T,0.1)
        # Calculate average error of weight vector on test data
        for example in test_D:
            actual = example[-1]
            features = example[:-1]
            guess = Perceptron.__guess__(features,weights)
            prediction = Perceptron.__sign__(guess)
            if actual != prediction:
                num_errors += 1
    error_rate = num_errors/(len(test_D)*ITERS)
    avg_error_rates.append(error_rate)
    print("{0}\t{1}".format(T,error_rate))

print('Weight vector with T = 10')
for weight in weights:
    print('{0:.2f},'.format(weight),end="")
print()
plt.plot(epochs,avg_error_rates,'b',epochs,avg_error_rates,'bo')
plt.ylim(0.0,max(avg_error_rates)+0.5*max(avg_error_rates))
plt.xlabel('Epochs')
plt.ylabel('Average Error')
plt.title('Average Error for Standard Perceptron')
plt.savefig('./figure_2a.pdf')
print('Please see file figure_2a.pdf')
print('Done with HW3 Part 2 Problem 2a!')
print()