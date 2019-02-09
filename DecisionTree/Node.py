class Node:
    def __init__(self,name):
        self.name = name
        self.branches = dict()

    def isLeaf(self):
        if len(self.branches) == 0:
            return True
        else:
            return False

'''n = Node("outlook")
n.branches = {'overcast':Node(''),'rainy':Node(''),'sunny':Node('')}
print(n.branches['overcast'].isLeaf())'''