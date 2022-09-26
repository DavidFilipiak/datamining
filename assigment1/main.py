import numpy as np
from sklearn.metrics import confusion_matrix

def tree_grow(x, y, nmin, minleaf, nfeat):
    return DecisionTree(x, y, nmin, minleaf, nfeat)

class DecisionTree:
    class Node:
        def __init__(self, attrs, labels):
            self.left = None
            self.right = None
            self.attrs = attrs
            self.labels = labels
            self.split_attr_index = -1
            self.split_value = 0

    def __init__(self, x, y, nmin, minleaf, nfeat):
        self.root = self.Node(x,y)
        self.nodeList = []
        self.nodeList.append(self.root)

        # iterate over every node in the node list (starting with root),
        # the nodes to the list are being added as more splits happen
        while len(self.nodeList) > 0:

            # take the first node of the node list
            node = self.nodeList.pop(0)

            # only attempt to split if number of observations is more than 'nmin' and the node is not pure
            if len(node.labels) >= nmin and impurity(node.labels) > 0:

                # get random attributes to try splitting on based on 'nfeat' (important for random forest & bagging)
                attrs_indexes = np.random.choice(np.arange(0, len(x[0])), size=nfeat, replace=False)

                split_attribute = 0
                best_impurity_reduction = 0
                best_split_value = 0

                # for every randomly selected attribute, determine which one provides the best split
                for i in attrs_indexes:
                    splitpoint, impurity_reduction = bestsplit(node.attrs[:, i], node.labels)
                    if impurity_reduction > best_impurity_reduction:
                        split_attribute = i
                        best_impurity_reduction = impurity_reduction
                        best_split_value = splitpoint

                # get indexes of values lower or equal AND greater than the best split attribute
                indexes_lower = np.arange(0, len(node.attrs[:, split_attribute]))[node.attrs[:, split_attribute] <= best_split_value]
                indexes_upper = np.delete(np.arange(0, len(node.attrs[:, split_attribute])), indexes_lower)

                # only continue if both branches of the best split have more observations than 'minleaf'
                if len(indexes_lower) >= minleaf and len(indexes_upper) >= minleaf:

                    # create a new child nodes to the current node, and add both to the node list
                    left = self.Node(node.attrs[indexes_lower], node.labels[indexes_lower])
                    right = self.Node(node.attrs[indexes_upper], node.labels[indexes_upper])
                    node.left, node.right = left, right
                    node.split_attr_index, node.split_value = split_attribute, best_split_value
                    self.nodeList.append(left)
                    self.nodeList.append(right)

    # recursively print the tree based on depth-first traversal
    def printNode(self, node, level):
        text = '\t'*level
        isLeafSign = "  "
        isLeaf = node.left is None and node.right is None
        if isLeaf:
            isLeafSign = "* "
        print(text + isLeafSign + "data: " + str(node.labels) + " split attribute: " + str(node.split_attr_index) + " impurity: " + str(impurity(node.labels)))
        if not isLeaf:
            self.printNode(node.left, level+1)
            self.printNode(node.right, level+1)

    def printTree(self):
        self.printNode(self.root, 0)

    def predict(self):
        pass

# predict the class of new case given a tree
def tree_pred(x,tr):
    y = []
    for i in x:
        # start in root node of tree
        node = tr.root
        # if not in a leaf node, check split condition and move to correct next node
        while not (node.left is None and node.right is None):
            case_value = i[node.split_attr_index]
            if case_value <= node.split_value:
                node = node.left
            else:
                node = node.right

        # assign new case to majority class of leaf node
        if np.mean(node.labels) <= 0.5:
            y.append(0)
        else:
            y.append(1)
    return y

array=np.array([1,0,1,1,1,0,0,1,1,0,1])
credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)

# gini-index impurity function: see presentation from lecture 37A, slide 21
def impurity(array):
    return (len(array[array == 0])/len(array)) * (len(array[array == 1])/len(array))

# redistribution error impurity function
def impurity_R(array, total_samples):
    return min(array[array == 0], array[array == 1]) / total_samples

def bestsplit(x,y):
    # sort the values of an attribute and determine splitpoints in between distinct values
    data_sorted = np.sort(np.unique(x))
    data_splitpoints = (data_sorted[0:len(data_sorted)-1] + data_sorted[1:len(data_sorted)]) / 2

    # the impurity of this node so that impurity reduction can be determined
    parent_impurity = impurity(y)
    best_point = 0
    best_impurity_reduction = 0

    # check impurity reduction for every splitpoint to see which is best
    for point in data_splitpoints:

        # get indexes of attributes with value lower or equal than the current splitpoint
        indexes_lower = np.arange(0, len(x))[x <= point]

        # impurity reduction function: see presentation from lecture 37A, slide 18
        imp_red = parent_impurity - ((len(indexes_lower) / len(x)) * impurity(y[indexes_lower]) + (
                    len(np.delete(y, indexes_lower)) / len(x)) * impurity(np.delete(y, indexes_lower)))

        if imp_red > best_impurity_reduction:
            best_impurity_reduction = imp_red
            best_point = point


    return best_point, best_impurity_reduction

# grows a tree on each of m bootstrap samples and returns list with these m trees
def tree_grow_b(x, y, nmin, minleaf, nfeat, m):

    tree_list = []

    for sample in range(m):
        tree = tree_grow(x, y, nmin, minleaf, nfeat)
        tree_list.append(tree)

    return tree_list

# applies tree_pred on x using each tree from tree_grow_b
# returns majority of the predictions as final prediction
def tree_pred_b(tr_list, x):

    predictions = []
    y = []

    for tree in tr_list:
        prediction = tree_pred(x,tree)
        predictions.append(prediction)

    for i in range(len(x)):
        i_predictions = []
        for prediction in predictions:
            i_predictions.append(prediction[i])

        if np.mean(i_predictions) <= 0.5:
            y.append(0)
        else:
            y.append(1)

    return y

#print(impurity(array))
#print(bestsplit(credit_data[:,3],credit_data[:,5]))
#print(bestsplit(np.array([10,10,10,20,20,30,30,40,40]),np.array([0,0,1,0,1,1,1,0,0]))) #test for homework 1, question 2

tree = tree_grow(credit_data[:,:5],credit_data[:,5],2,1,len(credit_data[0]) - 1)
#tree.printTree()

#print(tree_pred(credit_data[:,:5],tree))



pima_data = np.genfromtxt('pima_indians.txt', delimiter=',', skip_header=False)
tree2 = tree_grow(pima_data[:,:8],pima_data[:,8],20,5,len(pima_data[0]) - 1)
predictions = np.array(tree_pred(pima_data[:,:8], tree2))
originals = pima_data[:,8]

#  Expected output:
#  444  56
#  54   214
print(confusion_matrix(originals, predictions))
