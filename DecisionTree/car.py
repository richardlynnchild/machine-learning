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

with open('./car_results.csv','w') as results_file:
    results_file.write('Depth,Train_ME,Train_Gini,Train_Entropy,Test_ME,' +
                        'Test_Gini,Test_Entropy\n')

for max_depth in range(1,7):
    me = DT.ID3(S,Columns,Attributes,Labels,DT.__majority_error__,max_depth,0)
    gini = DT.ID3(S,Columns,Attributes,Labels,DT.__gini__,max_depth,0)
    entropy = DT.ID3(S,Columns,Attributes,Labels,DT.__entropy__,max_depth,0)

    train_me_success = 0
    train_gini_success = 0
    train_entropy_success = 0
    train_total = 0

    test_me_success = 0
    test_gini_success = 0
    test_entropy_success = 0
    test_total = 0

    with open('./car/train.csv','r') as test_file:
        for line in test_file:
            example = line.strip().split(',')
            if DT.Predict(example,me,Columns):
                train_me_success += 1
            if DT.Predict(example,gini,Columns):
                train_gini_success += 1
            if DT.Predict(example,entropy,Columns):
                train_entropy_success += 1
            train_total += 1

    with open('./car/test.csv','r') as test_file:
        for line in test_file:
            example = line.strip().split(',')
            if DT.Predict(example,me,Columns):
                test_me_success += 1
            if DT.Predict(example,gini,Columns):
                test_gini_success += 1
            if DT.Predict(example,entropy,Columns):
                test_entropy_success += 1
            test_total += 1

    train_me_er = 1-(train_me_success/train_total)
    train_gini_er = 1-(train_gini_success/train_total)
    train_entropy_er = 1-(train_entropy_success/train_total)

    test_me_er = 1-(test_me_success/test_total)
    test_gini_er = 1-(test_gini_success/test_total)
    test_entropy_er = 1-(test_entropy_success/test_total)

    with open('./car_results.csv','a') as results_file:
        results_file.write('{0},{1:0.3f},{2:0.3f},{3:0.3f},{4:0.3f},{5:0.3f},{6:0.3f}\n'.format(
            max_depth,train_me_er,train_gini_er,train_entropy_er,
            test_me_er,test_gini_er,test_entropy_er))


