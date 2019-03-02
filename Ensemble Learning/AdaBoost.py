'''
AdaBoost algorithm
gini = DT.ID3(S_train,Columns,Attributes,Labels,DT.__gini__,max_depth,0)
'''
import DecisionTree as DT
import math

'''
Use __gini__, max_depth=1,current_depth=0
'''
def AdaBoost(T,S,Columns,Attributes,Labels):
    D = [1/len(S) for x in range(len(S))]
    learners = []
    votes = []
    for _ in range(T):
        hypothesis = DT.ID3(S,Columns,Attributes,Labels,D)
        learners.append(hypothesis)
        error = __error__(hypothesis,S,Columns,D)
        vote = 0.5*math.log(((1-error)/error))
        votes.append(vote)

        for i,example in enumerate(S):
            prediction = DT.Predict(example,hypothesis,Columns)
            if prediction is True:
                D[i] *= math.exp(-vote)
            else:
                D[i] *= math.exp(vote)

        norm_constant = sum(D)
        D = [x/norm_constant for x in D]
    return learners,votes

def __error__(hypothesis,S,Columns,D):
    error = 0.0
    for i,example in enumerate(S):
        prediction = DT.Predict(example,hypothesis,Columns)
        if prediction is False:
            error += D[i]
    return error/len(S)
