import numpy as np
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
import time


# grow a DecisionTree from the input data, and return the tree object
# x -> observation matrix on which the tree should be grown
# y -> array of classification labels, each corresponding to one observation
# nmin -> number of observations that a node must contain at least for it to be allowed to be split
# minleaf -> the minimum number of observations required for a leaf node
# nfeat -> the number of features that should be considered for each split
def tree_grow(x, y, nmin, minleaf, nfeat):
    # input parameters are handed over to the DecisionTree constructor, that contains the logic for tree growing
    return DecisionTree(x, y, nmin, minleaf, nfeat)


# predict the class of new case given a tree
# x -> the table of observations to be predicted
# tr -> a trained DecisionTree object on which we predict the observations
def tree_pred(x,tr):
    y = []
    for i in x:
        # start in root node of tree
        node = tr.root
        # if not in a leaf node, check split condition and move to correct next node
        while not node.is_leaf():
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


# grows a tree on each of m bootstrap samples and returns list with these m trees
# m -> the number of bootstrap sampled trees to be grown
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


# applies tree_pred on x using each tree from tree_grow_b and returns majority of the predictions as final prediction
# tr_list -> a list of decision tree objects returned from tree_grow_b function
# x -> observation that should be predicted on each of the trees
def tree_pred_b(tr_list, x):

    predictions = []
    y = []

    # predict the observations using every tree from the list and add the predictions of that tree to the list
    for tree in tr_list:
        prediction = tree_pred(x,tree)
        predictions.append(prediction)

    # take the majority prediction class from every tree for every attribute
    for i in range(len(x)):
        i_predictions = []
        for prediction in predictions:
            i_predictions.append(prediction[i])

        if np.mean(i_predictions) <= 0.5:
            y.append(0)
        else:
            y.append(1)

    return y


# gini-index impurity function
# array -> the array of binary labels {0,1} on which the impurity is calculated
def impurity(array):
    return (len(array[array == 0])/len(array)) * (len(array[array == 1])/len(array))


# function to find the optimal split from the provided data,
# returns the value on which the optimal split was performed and calculated impurity reduction from the optimal split
# x -> an array of observation values from a single column / attribute
# y -> an array of labels corresponding to the observations
# minleaf -> the minimum number of observations required for a leaf node
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

        # only consider the split to be suitable if it satisfies the minleaf constraint as well as provides better impurity reduction
        if len(indexes_lower) >= minleaf and len(indexes_upper) >= minleaf and imp_red > best_impurity_reduction:
            best_impurity_reduction = imp_red
            best_point = point

    return best_point, best_impurity_reduction


# the DecisionTree class that creates decision tree objects
class DecisionTree:
    # nested class Node that represents nodes on the decision tree
    class Node:
        def __init__(self, attrs, labels):
            self.left = None
            self.right = None
            self.attrs = attrs
            self.labels = labels
            self.split_attr_index = -1
            self.split_value = 0

        def is_leaf(self):
            return self.left is None and self.right is None

    # DecisionTree constuctor that creates the tree from provided parameters
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

                # get random attributes to try splitting on based on 'nfeat' (important for random forest)
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

    # recursively print the tree based on depth-first inorder traversal
    # node -> the node that should be displayed
    # level -> the depth level of the current node
    # max_levels -> the maximum depth of the tree that should be displayed. Negative number displays the entire tree.
    def printNode(self, node, level, max_levels):
        if node is not None and (level <= max_levels or max_levels < 0):
            text = '\t'*level*2
            is_leaf_sign = "  "
            if node.is_leaf():
                is_leaf_sign = "* "

            self.printNode(node.left, level+1, max_levels)
            print(text + is_leaf_sign + "LEVEL " + str(level) + ": 0 = [" + str(len(node.labels[node.labels == 0])) + "], 1 = [" + str(len(node.labels[node.labels == 1])) + "], impurity: " + str(impurity(node.labels)))
            self.printNode(node.right, level+1, max_levels)

    # function that prints the decision tree to the console.
    # max_levels -> the maximum depth of the tree that should be displayed. Negative number displays the entire tree.
    def printTree(self, max_levels=-1):
        self.printNode(self.root, 0, max_levels)


###
###
###
print("\n-----------\n ANALYSIS\n-----------")
###
###
###


# calculate and display accuracy, precision and recall from provided confusion matrix
def calculate_metrics(confusion_matrix):
    ##Acuracy
    acc = (confusion_matrix[1,1] + confusion_matrix[0,0]) / confusion_matrix.sum()
    print("Accuracy: ", acc)

    ## Precision
    prc = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1])
    print("Precision: ", prc)

    ## Recall
    rcall = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0])
    print("Recall: ", rcall)

# pefrom McNemar statistical test on provided ...
# predictions_a -> ...
# predictions_b -> ...
# classification -> ...
def performMcNemar(predictions_a, predictions_b, classification):
    i = 0
    no_no = 0
    no_yes = 0
    yes_no = 0
    yes_yes = 0

    while i < len(predictions_a):
        if classification[i] == predictions_a[i] and predictions_a[i] == predictions_b[i]:
            yes_yes +=1
        if classification[i] != predictions_a[i] and classification[i] != predictions_b[i]:
            no_no +=1
        if classification[i] == predictions_a[i] and classification[i] != predictions_b[i]:
            yes_no +=1
        if classification[i] == predictions_b[i] and classification[i]  != predictions_a[i]:
            no_yes +=1
        i+=1
    ## define contingency table
    table = [[yes_yes, yes_no],
    		 [no_yes, no_no]]
    print(table)
    result = mcnemar(table, exact=True)
    print('statistic=%.5f, p-value=%.35f' % (result.statistic, result.pvalue))
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
print("\n TREE")
start = time.time()
tree_eclipse = tree_grow(eclipse_train_x, eclipse_train_y, 15, 5, 41)
print("Time to grow: " + str(time.time() - start) + " seconds")

tree_eclipse.printTree(max_levels=3)

start = time.time()
predictions_tree = np.array(tree_pred(eclipse_test_x, tree_eclipse))
print("Time to predict: " + str(time.time() - start) + " seconds")

cm_tree = confusion_matrix(eclipse_test_y, predictions_tree)

print(cm_tree)
calculate_metrics(cm_tree)

## Bagging analysis
print("\n BAGGING")
start = time.time()
bagging_eclipse = tree_grow_b(eclipse_train_x, eclipse_train_y, 15, 5, 41, 100)
print("Time to grow: " + str(time.time() - start) + " seconds")

start = time.time()
predictions_bagging = np.array(tree_pred_b(bagging_eclipse, eclipse_test_x))
print("Time to predict: " + str(time.time() - start) + " seconds")

cm_bagging = confusion_matrix(eclipse_test_y, predictions_bagging)

print(cm_bagging)
calculate_metrics(cm_bagging)

## Random forest analysis
print("\n RANDOM FOREST")
start = time.time()
rf_eclipse = tree_grow_b(eclipse_train_x, eclipse_train_y, 15, 5, 6, 100)
print("Time to grow: " + str(time.time() - start) + " seconds")

start = time.time()
predictions_rf = np.array(tree_pred_b(rf_eclipse, eclipse_test_x))
print("Time to predict: " + str(time.time() - start) + " seconds")

cm_rf = confusion_matrix(eclipse_test_y, predictions_rf)

print(cm_rf)
calculate_metrics(cm_rf)

## Statistical Test - McNemar
print("\n STATISTICAL TESTS")
print("McNemar - Single Tree vs Bagging Forest")
performMcNemar(predictions_tree, predictions_bagging, eclipse_test_y)
print("\n")
print("McNemar - Bagging Forest vs Random Forest")
performMcNemar(predictions_bagging, predictions_rf,eclipse_test_y)
print("\n")
print("McNemar - Single Tree vs Random Forest")
performMcNemar(predictions_tree, predictions_rf,eclipse_test_y)