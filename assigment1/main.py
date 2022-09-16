import numpy as np

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

    return best_point


print(impurity(array))
print(bestsplit(credit_data[:,3],credit_data[:,5]))