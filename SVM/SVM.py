'''
Author: Richard Child
Date:	April 5, 2019

Machine Learning - CS 5350 - University of Utah

SVM algorithm implementations

svm_primal - 	implements a SVM in the primal domain
				using stochastic sub-gradient descent.

svm_dual - 		implements a SVM in the dual domain.
'''
import numpy as np 
import copy
from random import shuffle  

'''
Params: S (dataset), T (number of epochs), C (hyperparameter constant),
gamma (initial value of learning rate)
Returns: w (weight vector used for future predictions)
Preconditions: the dataset S is assumed to be of the form [x,y] where
x is vector of features, and y is the label. The label must be [-1,1].
Also it is assumed the bias term has been folded into x (x->[x,1])
'''
def svm_primal(S,T,C,gamma_0,schedule):
	if len(S) < 1:
		raise Exception('Dataset is empty')
	w = np.zeros(len(S[0])-1)
	for epoch in range(T):
		__sub_gradient_descent__(w,S,epoch,C,gamma_0,schedule)
	return w

'''
Params: w (weight vector), S (dataset), C (hyperparameter constant)
Returns: w (newly updated weight vector)
Preconditions: the dataset S is of form [x,y], where x is feature
vector and y is label. The label must be [-1,1].
'''

def __sub_gradient_descent__(w,S,T,C,gamma_0,schedule):
	t = T*len(S)
	N = len(S)
	shuffle(S)
	for example in S:
		y = example[-1]
		x = example[:-1]
		hinge_loss = y*np.matmul(w,x)
		#gamma_t = schedule(t,C,gamma_0)
		gamma_t = (gamma_0/(1+(gamma_0*t/C)))
		if hinge_loss <= 1.0:
			w = (1-gamma_t)*w
			scalar = gamma_t*C*N*y
			w += np.multiply(x,scalar)
		else :
			w = (1-gamma_t)*w
		t += 1
	return


def __first_learning_schedule__(t,C,gamma_0):
	return (gamma_0/(1+(gamma_0*t/C)))


def __second_learning_schedule__(t,C,gamma_0):
	return gamma_0/(1+t)




