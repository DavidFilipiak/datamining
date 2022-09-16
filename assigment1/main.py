import numpy as np

array=np.array([1,0,1,1,1,0,0,1,1,0,1])
credit_data = np.genfromtxt('D:/UU/Data Mining/assigment1/credit.txt', delimiter=',', skip_header=True)

def impurity(array):
    return (len(array[array == 0])/len(array)) * (len(array[array == 1])/len(array))

def bestsplit(x,y):
    data_sorted = np.sort(np.unique(x))
    data_splitpoints = (data_sorted[0:len(data_sorted)-1] + data_sorted[1:len(data_sorted)]) / 2

    best_point = data_splitpoints[0]
    best_impurity = 1
    for point in data_splitpoints:
        indexes_lower = np.arange(0, len(x))[x <= point]
        #indexes_higher = np.arange(0, len(x))[x[:, 0] > point]
        imp = impurity(y[indexes_lower])
        print(point, x[indexes_lower], y[indexes_lower],imp)
        if imp < best_impurity:
            best_impurity = impurity(y[indexes_lower])
            best_point = point

    return best_point


print(bestsplit(credit_data[:,3],credit_data[:,4]))






print(impurity(array))