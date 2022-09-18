import numpy as np
import random

class DecisionTree:
    class Node:
        def __init__(self, attrs, labels):
            self.left = None
            self.right = None
            self.attrs = attrs
            self.labels = labels

    def __init__(self, x, y, nmin, minleaf, nfeat):
        self.root = self.Node(x,y)
        self.nodeList = []
        self.nodeList.append(self.root)

        while len(self.nodeList) > 0:
            node = self.nodeList.pop(0)
            print(node.attrs)
            if len(node.labels) >= nmin and impurity(node.labels) > 0:

                attrsIndexes = [i for i in range(len(x[0]))]
                while len(attrsIndexes) > nfeat:
                    number = random.randint(0,len(attrsIndexes))
                    attrsIndexes.pop(number)

                split_attribute = 0
                best_impurity_reduction = 0
                best_split_value = 0
                for i in attrsIndexes:
                    splitpoint, impurity_reduction = bestsplit(x[:,i], y)
                    if splitpoint > best_impurity_reduction:
                        split_attribute = i
                        best_impurity_reduction = impurity_reduction
                        best_split_value = splitpoint

                lower_split = x[x[:,split_attribute] <= best_split_value]
                upper_split = x[x[:,split_attribute] > best_split_value]

                if len(lower_split) >= minleaf and len(upper_split) >= minleaf:
                    left = self.Node(lower_split,y[np.arange(0,len(y))[x[:,split_attribute] <= best_split_value]])
                    right = self.Node(upper_split,y[np.arange(0,len(y))[x[:,split_attribute] > best_split_value]])
                    node.left, node.right = left, right
                    self.nodeList.append(left)
                    self.nodeList.append(right)


    def printNode(self, node, level):
        print("\t"*level + "data: "+node.attrs)
        if node.left is not None:
            self.printNode(node.left, level+1)
        if node.right is not None:
            self.printNode(node.right, level+1)

    def __str__(self):
        self.printNode(self.root, 0)






array=np.array([1,0,1,1,1,0,0,1,1,0,1])
credit_data = np.genfromtxt('D:/UU/Data Mining/datamining/assigment1/credit.txt', delimiter=',', skip_header=True)

#gini-index impurity function: see presentation from lecture 37A, slide 21
def impurity(array):
    return (len(array[array == 0])/len(array)) * (len(array[array == 1])/len(array))

def bestsplit(x,y):
    data_sorted = np.sort(np.unique(x))
    data_splitpoints = (data_sorted[0:len(data_sorted)-1] + data_sorted[1:len(data_sorted)]) / 2

    parent_impurity = impurity(y)

    best_point = data_splitpoints[0]
    best_impurity_reduction = 0
    for point in data_splitpoints:
        indexes_lower = np.arange(0, len(x))[x <= point]
        #indexes_higher = np.arange(0, len(x))[x > point]
        #y[indexes_higher] == np.delete(y,indexes_lower)
        #impurity reduction function: see presentation from lecture 37A, slide 18
        imp_red = parent_impurity - ((len(indexes_lower)/len(x)) * impurity(y[indexes_lower]) + (len(np.delete(y,indexes_lower))/len(x)) * impurity(np.delete(y,indexes_lower)))
        #print(point, indexes_lower, x[indexes_lower], y[indexes_lower],imp_red)
        if imp_red > best_impurity_reduction:
            best_impurity_reduction = imp_red
            best_point = point

    return best_point, best_impurity_reduction


print(impurity(array))
print(bestsplit(credit_data[:,3],credit_data[:,5]))
print(bestsplit(np.array([10,10,10,20,20,30,30,40,40]),np.array([0,0,1,0,1,1,1,0,0]))) #test for homework 1, question 2
print(DecisionTree(credit_data[:,:4],credit_data[:,5],2,2,len(credit_data[:,5])))