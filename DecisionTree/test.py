from DecisionTree import DecisionTree
import DecisionTree as DT

S = []

Columns = ['buying','maint','doors','persons','lug_boot','safety','label']

Attributes = {'buying':['vhigh','high','med','low'],
                'maint':['vhigh','high','med','low'],
                'doors':['2','3','4','5more'],
                'persons':['2','4','more'],
                'lug_boot':['small','med','big'],
                'safety':['low','med','high']}

with open('./car/train.csv', 'r') as train_file:
    for line in train_file:
        terms = line.strip().split(',')
        S.append(terms)

Labels = set()
for example in S:
    Labels.add(example[len(example)-1])

root = DT.ID3(S,Columns,Attributes,Labels,DT.__gini__,1,0)
tree = DecisionTree(root)
print(tree.root.name)
