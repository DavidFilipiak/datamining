import numpy as np

def tree_grow(x, y, nmin, minleaf, nfeat):
    return DecisionTree(x, y, nmin, minleaf, nfeat)

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
            if len(node.labels) >= nmin and impurity(node.labels) > 0:

                attrsIndexes = np.random.choice(np.arange(0, len(x[0])), size=nfeat, replace=False)

                split_attribute = 0
                best_impurity_reduction = 0
                best_split_value = 0
                for i in attrsIndexes:
                    splitpoint, impurity_reduction = bestsplit(node.attrs[:,i], node.labels)
                    if impurity_reduction > best_impurity_reduction:
                        split_attribute = i
                        best_impurity_reduction = impurity_reduction
                        best_split_value = splitpoint

                lower_split = node.attrs[node.attrs[:,split_attribute] <= best_split_value]
                upper_split = node.attrs[node.attrs[:,split_attribute] > best_split_value]

                if len(lower_split) >= minleaf and len(upper_split) >= minleaf:
                    left = self.Node(lower_split,node.labels[np.arange(0,len(node.labels))[node.attrs[:,split_attribute] <= best_split_value]])
                    right = self.Node(upper_split,node.labels[np.arange(0,len(node.labels))[node.attrs[:,split_attribute] > best_split_value]])
                    node.left, node.right = left, right
                    self.nodeList.append(left)
                    self.nodeList.append(right)


    def printNode(self, node, level):
        text = '\t'*level
        isLeafSign = "  "
        isLeaf = node.left is None and node.right is None
        if isLeaf:
            isLeafSign = "* "
        print(text + isLeafSign + "data: " + str(node.labels) + " impurity: " + str(impurity(node.labels)))
        if not isLeaf:
            self.printNode(node.left, level+1)
            self.printNode(node.right, level+1)

    def printTree(self):
        self.printNode(self.root, 0)

    def predict(self):
        pass






array=np.array([1,0,1,1,1,0,0,1,1,0,1])
credit_data = np.genfromtxt('D:/UU/Data Mining/datamining/assigment1/credit.txt', delimiter=',', skip_header=True)

#gini-index impurity function: see presentation from lecture 37A, slide 21
def impurity(array):
    return (len(array[array == 0])/len(array)) * (len(array[array == 1])/len(array))

def bestsplit(x,y):
    data_sorted = np.sort(np.unique(x))
    data_splitpoints = (data_sorted[0:len(data_sorted)-1] + data_sorted[1:len(data_sorted)]) / 2

    parent_impurity = impurity(y)
    best_point = 0
    best_impurity_reduction = 0
    if len(data_splitpoints) > 0:
        for point in data_splitpoints:
            indexes_lower = np.arange(0, len(x))[x <= point]
            # indexes_higher = np.arange(0, len(x))[x > point]
            # y[indexes_higher] == np.delete(y,indexes_lower)
            # impurity reduction function: see presentation from lecture 37A, slide 18
            imp_red = parent_impurity - ((len(indexes_lower) / len(x)) * impurity(y[indexes_lower]) + (
                        len(np.delete(y, indexes_lower)) / len(x)) * impurity(np.delete(y, indexes_lower)))
            # print(point, indexes_lower, x[indexes_lower], y[indexes_lower],imp_red)
            if imp_red > best_impurity_reduction:
                best_impurity_reduction = imp_red
                best_point = point


    return best_point, best_impurity_reduction


print(impurity(array))
print(bestsplit(credit_data[:,3],credit_data[:,5]))
print(bestsplit(np.array([10,10,10,20,20,30,30,40,40]),np.array([0,0,1,0,1,1,1,0,0]))) #test for homework 1, question 2

tree = tree_grow(credit_data[:,:5],credit_data[:,5],2,1,len(credit_data[0]) - 1)
tree.printTree()

pima_data = np.genfromtxt('D:/UU/Data Mining/datamining/assigment1/pima_indians.txt', delimiter=',', skip_header=False)
tree2 = tree_grow(pima_data[:,:8],pima_data[:,8],20,5,len(pima_data[0]) - 1)
tree2.printTree()