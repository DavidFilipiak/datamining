import numpy as np
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

import time

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
                    splitpoint, impurity_reduction = bestsplit(node.attrs[:, i], node.labels, minleaf)
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

def bestsplit(x,y,minleaf):
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
        indexes_upper = np.delete(np.arange(0, len(y)), indexes_lower)

        # impurity reduction function: see presentation from lecture 37A, slide 18
        imp_red = parent_impurity - ((len(indexes_lower) / len(x)) * impurity(y[indexes_lower]) + (len(indexes_upper) / len(x)) * impurity(y[indexes_upper]))

        if len(indexes_lower) >= minleaf and len(indexes_upper) >= minleaf and imp_red > best_impurity_reduction:
            best_impurity_reduction = imp_red
            best_point = point


    return best_point, best_impurity_reduction

# grows a tree on each of m bootstrap samples and returns list with these m trees
def tree_grow_b(x, y, nmin, minleaf, nfeat, m):

    tree_list = []
    index = 0
    for sample in range(m):
        print(f"\rGrowing trees: {index + 1}/{m}", end='')
        # perform bootstrap sampling on x
        boostrap_sample_indexes = np.random.choice(np.arange(0, len(y)), size=len(y), replace=True)
        tree = tree_grow(x[boostrap_sample_indexes], y, nmin, minleaf, nfeat)
        tree_list.append(tree)
        index += 1

    print()
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

#tree = tree_grow_b(credit_data[:,:5],credit_data[:,5],2,1,len(credit_data[0]) - 1,1)
#tree.printTree()

#print(tree_pred(credit_data[:,:5],tree))



test_data = np.genfromtxt('pima_indians.txt', delimiter=',', skip_header=False) # pima_indians
#test_data = np.genfromtxt('eclipse-metrics-packages-2.0.csv', delimiter=';', skip_header=True) # test set 1
size = len(test_data[0]) - 1
start = time.time()
tree2 = tree_grow_b(test_data[:,:size],test_data[:,size],15,5,size,1)
print("Tree grow:", time.time() - start, "seconds")
#tree2.printTree()
start = time.time()
predictions = np.array(tree_pred_b(tree2, test_data[:,:size]))
print("Tree pred:", time.time() - start, "seconds")
originals = test_data[:,size]

#  Expected output:
#  444  56
#  54   214
conf_matrix = confusion_matrix(originals, predictions)
accuracy = (conf_matrix[1,1] + conf_matrix[0,0]) / conf_matrix.sum()
precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1])
recall = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
print("Confusion Matrix:", conf_matrix, "Accuracy:", accuracy, "Precision:", precision, "Recall", recall, sep="\n")

###
###
###
print("-----------", " ANALYSIS", "-----------", sep="\n")
###
###
###

def calculateMetrics(confusion_matrix):
    ##Acuracy
    acc = (confusion_matrix[1,1] + confusion_matrix[0,0]) / confusion_matrix.sum()
    print("Accuracy: ", acc)

    ## Precision
    prc = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1])
    print("Precision: ", prc)

    ## Recall
    rcall = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0])
    print("Recall: ", rcall)

def performMcNemar(predictions_a, predictions_b, classification):
    i = 0
    no_no = 0
    no_yes = 0
    yes_no = 0
    yes_yes = 0

    while i < len(predictions_a):
        if classification[i] == predictions_a[i] == predictions_b[i]:
            yes_yes +=1
        if classification[i] != (predictions_a[i] == predictions_b[i]):
            no_no +=1
        if classification[i] == predictions_a[i] != predictions_b[i]:
            yes_no +=1
        if classification[i] == predictions_b[i] != predictions_a[i]:
            no_yes +=1
        i+=1
    ## define contingency table
    table = [[yes_yes, yes_no],
    		 [no_yes, no_no]]
    print(table)
    result = mcnemar(table, exact=True)
    print('statistic=%.5f, p-value=%.35' % (result.statistic, result.pvalue))
    # interpret the p-value 
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
    	print('Different proportions of errors (reject H0)')

## Analysis
## File Reading
eclipse_train_data = np.genfromtxt('eclipse-metrics-packages-2.0.csv', delimiter=';', encoding="utf8")
eclipse_test_data = np.genfromtxt('eclipse-metrics-packages-3.0.csv', delimiter=';', encoding="utf8")

# Training set
eclipse_train_x = eclipse_train_data[1:,:41]
eclipse_train_y = eclipse_train_data[1:,41]

# Testing set
eclipse_test_x = eclipse_test_data[1:,:41]
eclipse_test_y = eclipse_test_data[1:,41]

## Tree analysis
tree_eclipse = tree_grow(eclipse_train_x, eclipse_train_y, 15, 5, 41)
tree_eclipse.printTree()

predictions_tree = np.array(tree_pred(eclipse_test_x, tree_eclipse))
cm_tree = confusion_matrix(eclipse_test_y, predictions_tree)

print("\n TREE")
print(cm_tree)
calculateMetrics(cm_tree)

## Bagging analysis
bagging_eclipse = tree_grow_b(eclipse_train_x, eclipse_train_y, 15, 5, 41, 100)

predictions_bagging = np.array(tree_pred_b(bagging_eclipse, eclipse_test_x))
cm_bagging = confusion_matrix(eclipse_test_y, predictions_bagging)

print("\n BAGGING")
print(cm_bagging)
calculateMetrics(cm_bagging)

## Random forest analysis
rf_eclipse = tree_grow_b(eclipse_train_x, eclipse_train_y, 15, 5, 6, 100)

predictions_rf = np.array(tree_pred_b(rf_eclipse, eclipse_test_x))
cm_rf = confusion_matrix(eclipse_test_y, predictions_rf)

print("\n RANDOM FOREST")
print(cm_rf)
calculateMetrics(cm_rf)

## Statistical Test - McNemar
print("McNemar - Single Tree vs Bagging Forest")
performMcNemar(predictions_tree, predictions_bagging, eclipse_test_y)
print("\n")
print("McNemar - Bagging Forest vs Random Forest")
performMcNemar(predictions_bagging, predictions_rf,eclipse_test_y)

print("McNemar - Single Tree vs Random Forest")
performMcNemar(predictions_tree, predictions_rf,eclipse_test_y)