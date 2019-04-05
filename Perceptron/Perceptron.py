from random import shuffle
import copy
'''
Standard Perceptron algorithm
Parameters: D (dataset), T (number of epochs), r (learning rate)
Returns: weight vector
'''
def perceptron(D,T,r):
    w = [0 for x in range(len(D[0])-1)]
    for _ in range(0,T):
        shuffle(D)
        for example in D:
            features = example[:-1]
            guess = __guess__(features,w)
            sign = __sign__(guess)
            error = example[-1]-sign
        for i,feature in enumerate(features):
            w[i] += r*error*feature
    return w

'''
Averaged Perceptron algorithm.
Take the average of all the weight vectors produced on
each example in every epoch.
Parameters: D (datatset), T (number of epochs), r (learning rate)
Returns: average weight vector
'''
def avg_perceptron(D,T,r):
    count = 0
    average = [x for x in range(1,len(D[0]))]
    weights = [0 for x in range(1,len(D[0]))]
    # Go through T times
    for _ in range(0,T):
        # Start by shuffling the order of the dataset
        shuffle(D)
        # Go through all the examples
        for example in D:
            features = example[:-1]
            guess = __guess__(features,weights)
            sign = __sign__(guess)
            error = example[-1]-sign
            # Update the weight vector on every example
            for i,feature in enumerate(features):
                weights[i] += r*error*feature
            # Update count on each example
            count += 1
            # Update the average vector on each example
            for i,_ in enumerate(weights):
                average[i] += weights[i]
    # After running through all epochs (T), take the average
    # of the aggregated weight vector (average vector) by
    # diving by the total count of examples (len(D)*T)
    for i,a in enumerate(average):
        average[i] = a/count          
    return average  

'''
Voted Perceptron algorithm
Parameters: D (dataset), T (number of epochs), r (learning rate)
Returns: Weight vectors and vote vector
'''
def vote_perceptron(D,T,r):
    weights_list = []
    w = [0 for x in range(len(D[0])-1)]
    weights_list.append(w)
    m = 0
    c = []
    c.append(1)
    for _ in range(0,T):
        for example in D:
            features = example[:-1]
            guess = __guess__(features,w)
            sign = __sign__(guess)
            error = example[-1]-sign
            if error == 0:
                c[m] += 1
            else:
                w_new = copy.copy(w)
                for i,feature in enumerate(features):
                    w_new[i] += r*error*feature
                weights_list.append(w_new)
                w = copy.copy(w_new)
                m += 1
                c.append(1)
    return weights_list,c

def __guess__(features,weights):
    guess = 0.0
    for i,feature in enumerate(features):
        guess += feature*weights[i]
    return guess

def __sign__(number):
    if number >= 0:
        return 1
    else:
        return -1

'''
Make a prediction using the Voted Perceptron weights and votes
Parameters: weights vector, vote vector, example features
Return: prediction (1 or -1)'''

def __vote_prediction__(weights_list,vote_vector,features):
    sum = 0.0
    for i,weights in enumerate(weights_list):
        sum += vote_vector[i]*__sign__(__guess__(features,weights))
    return __sign__(sum)
