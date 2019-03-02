'''
Author: Richard Child
CS 5350 - Assignment 2

This is a script for Section 2, Question 4a.
A Batch Gradient Descent is performed on training data, a weight vector
is calculated. Then the vector is applied to test data and the cost
is calculated. More results are saved in the file concrete_4a_results.txt which
included Cost vs. Step data during the Batch Gradient Descent. Also a figure is
supplied as concrete_4a_fig.pdf

'''

import LinearRegression as LR 

S_train = []
print('Loading training data from concrete/train.csv...')
with open('./concrete/train.csv', 'r') as train_file:
    for line in train_file:
        terms = line.strip().split(',')
        terms_float = [float(t) for t in terms]
        S_train.append(terms_float)

print('About to perform linear regression using Batch Gradient descent...')
print('Will find optimal learning rate (r) to get convergence...')
print('This may take a few minutes...')
w,r = LR.BGD(S_train)
print('Done!')
print('See file concrete_4a_results.txt for cost vs. step statistics')
print(str(w))
print('Now we will use weight vector on test dataset and calculate cost...')
print('Reading in test data from concrete/test.csv...')
S_test = []
with open('./concrete/test.csv','r') as test_file:
    for line in test_file:
        terms = line.strip().split(',')
        terms_float = [float(t) for t in terms]
        S_test.append(terms_float)

print('The cost is: ' + str(LR.__cost__(S_test,w)))