import numpy as np  
import copy
import math
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')

'''
Calculate the cost (or loss) of a weight vector using LMS.
Params: S (the dataset), w (weight vector), b (bias term)
Returns: cost (real number)
'''
def __cost__(S,w):
    cost = 0.0
    for example in S:
        outcome = example[-1]
        features = example[:-1]
        features.append(1.0)
        error = outcome - np.matmul(w,features)
        cost = cost + (error**2)
    cost = cost/2
    return cost

'''
Calculate the gradient of the current cost function.
Return: gradient vector (same length as w)
'''
def __gradient__(S,w):
    r,c = w.shape
    gradient = np.zeros((1,c))
    for i in range(c):
        #print('Calculating _g_w_' + str(i))
        val = 0.0
        m = 0
        for example in S:
            #print('The example ' + str(example))
            outcome = example[-1]
            features = example[:-1]
            features.append(1.0)
            features = np.array(features)
            #print('Length of w: ' + str(len(w)))
            #print('Length of features: ' + str(len(features)))
            #print(w)
            #print(features)
            #print(np.matmul(w,features))
            val -= (outcome - np.matmul(w,features))*features[i]
            #print('The error is: ' + str(error))
            m+=1
        np.put(gradient, i, val/m)
    return gradient

'''
Batch Gradient Descent (BGD)
Use Least Mean Square linear regression to return a Weight vector and bias
term that can be applied to an input vector in order to predict an outcome.
Parameters: S, dataset which includes outcomes as last value in each row.
Return: (w,r), the weight vector and learning rate.
'''
def BGD(S):
    rates = [x*.01 for x in range(100,0,-5)]
    for r in rates:
        w,converged = __bgd__(S,r)
        if converged:

            f = open('./concrete_4a_results.txt','r')
            temp = f.read()
            f.close()

            fileResults = open('./concrete_4a_results.txt','w')
            fileResults.write('Weight vector: ' + str(w) + '\n')
            fileResults.write('Learning rate: ' + str(r) + '\n')
            fileResults.write(temp)
            fileResults.close()
            return w,r
    return w,r

'''
Helper function where 'r' can be specified.
Returns: (w,converged), weight vector and Boolean if Converged or not
'''
def __bgd__(S,r):
    
    print('Trying batch gradient descent with learning rate: {0:.2f} ... '.format(r), end='')

    # Set up some limits and variables
    MAX_ITERS = 10000
    tol = 1.0e-6    
    costs = []
    converged = True
    
    # Initial step in gradient descent
    w = np.zeros((1,len(S[0])))
    costs.append(__cost__(S,w))
    gradient = __gradient__(S,w)
    w_new = w - r*gradient
    costs.append(__cost__(S,w_new))
    w_diff = np.linalg.norm(w_new-w)

    # Iterate until convergence, or until reach MAX_ITERS
    iter = 2
    while w_diff > tol and iter < MAX_ITERS and not math.isinf(w_diff):
        w = w_new
        gradient = __gradient__(S,w)
        w_new = w - r*gradient
        costs.append(__cost__(S,w_new))
        w_diff = np.linalg.norm(w_new-w)
        iter += 1

    w = w_new

    # If did not converge, return
    if w_diff > tol:
        print('did not converge.')
        return w,False

    # Write costs to file
    with open('./concrete_4a_results.txt','w') as fileResults:
        fileResults.write('Step\tCost\n')
        for index,cost in enumerate(costs):
            fileResults.write(str(index)+ '\t' + str(cost) + '\n')

    # Create figure and save to file
    plt.semilogx([y for y in range(len(costs))], costs)
    plt.title('Linear regression cost value using Batch Gradient Descent')
    plt.ylabel('Cost')
    plt.xlabel('Step')
    plt.savefig('./concrete_4a_fig.pdf')

    print('converged!')
    return w,converged


