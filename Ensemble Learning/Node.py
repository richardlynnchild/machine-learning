'''
Author: Richard Child

Simple Node object to help implement Decision Tree.
'''

class Node:
    def __init__(self,name):
        self.name = name
        self.branches = dict()

    def isLeaf(self):
        if len(self.branches) == 0:
            return True
        else:
            return False
