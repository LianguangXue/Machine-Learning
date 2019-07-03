# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:14:34 2019
HW3 FOR DATAMINING
@author: Lianguang Xue
"""
import pandas as pd
import numpy as np
from collections import Counter
from scipy.io import arff
from sklearn.model_selection import LeaveOneOut
#2. Decision Tree 
def entropy(info):
    n = len(info)
    prob = [i/sum(info) for i in info] 
    H = 0
    for i in range(n):
        if prob[i]==0:
            return(0)
            break;
        else:
            H = H - prob[i]*np.log2(prob[i]) 
    return(H)
#3. Filter Method
#Define vote function
def vote(label,distance):
    count = Counter(label).most_common()
    if len(count) == 1:
        return count[0][0]
    elif count[0][1] > count[1][1]:
        return count[0][0]
    elif count[1][1] > count[2][1]:
        temp0 = 0
        index0 = [i for i, x in enumerate(label) if x==count[0][0]]
        for a in index0:
            temp0 += distance[a]
        temp1 = 0
        index1 = [i for i, x in enumerate(label) if x==count[1][0]]
        for a in index1:
            temp1 += distance[a]
        ss = [temp0,temp1].index(max([temp0,temp1]))
        return count[ss][0]
    else:
        temp0 = 0
        index0 = [i for i, x in enumerate(label) if x==count[0][0]]
        for a in index0:
            temp0 += distance[a]
        temp1 = 0
        index1 = [i for i, x in enumerate(label) if x==count[1][0]]
        for a in index1:
            temp1 += distance[a]
        temp2 = 0
        index2 = [i for i, x in enumerate(label) if x==count[2][0]]
        for a in index2:
            temp2 += distance[a]
        ss = [temp0,temp1,temp2].index(max([temp0,temp1,temp2]))
        return count[ss][0]
#Define KNN function
def KNN_predict(data_train,data_test,k):  
    classifer = {}
    class_predict = []
    p = data_train.shape[1]-1
    x_train = data_train.iloc[:,:p]
    y_train = data_train.iloc[:,p]
    n = len(data_test)
    for i in range(n):
        temp = np.sum((x_train-data_test.iloc[i,:])**2,axis=1)
        sort_temp = sorted(enumerate(temp), key=lambda x:x[1])
        index = [x[0] for x in sort_temp]
        index_distance = [x[1] for x in sort_temp][0:k]
        class_label = []
        for a in index[0:k]:
            class_label.append(np.array(y_train)[a])
        classifer[i] = class_label
        class_predict.append(vote(classifer[i],index_distance))
    return class_predict
#load data
raw_data = arff.loadarff('veh-prime.arff')
data = pd.DataFrame(raw_data[0])
#Make the class labels numeric (set “noncar”=0 and “car”=1) 
data['CLASS'] = data.apply(lambda row: 1 if row['CLASS'] == b'car' else 0,axis =1)
data.head() #look first five rows
data.tail() #look last five rows
#Define pearson function
#Pearson Correlation Coefficient Pseudocode
##Cov(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY]-E[X]E[Y]
##Sx = sqrt(E[X^2]-E[X]^2)
def pearson(x,y):
    N = len(x)
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    mean_x = 0
    mean_y = 0
    for i in range(N):
        sum_sq_x += x[i] * x[i]
        sum_sq_y += y[i] * y[i]
        sum_coproduct += x[i] * y[i]
        mean_x += x[i]
        mean_y += y[i]
    mean_x = mean_x / N
    mean_y = mean_y / N
    pop_sd_x = np.sqrt((sum_sq_x / N) - (mean_x * mean_x))
    pop_sd_y = np.sqrt((sum_sq_y / N) - (mean_y * mean_y))
    cov_x_y = (sum_coproduct / N) - (mean_x * mean_y)
    correlation = cov_x_y / (pop_sd_x * pop_sd_y)
    return correlation
#(1)
# calculate the Pearson Correlation Coeﬃcient (PCC) of each feature with the numeric class label
p = data.shape[1]-1
r = []
for i in range(p):
    r.append(pearson(data.iloc[:,i],data.iloc[:,p]))
print(r)
#get the absolute value of r 
r_abs = [abs(i) for i in r]
r_dataframe = pd.DataFrame(r_abs, index = data.columns[:36])
r_dataframe = r_dataframe.sort_values(by=0,ascending=False)
#List the features from highest |r| (the absolute value of r) to lowest, along with their |r| values
print(r_dataframe)
#(2)
#call the LeaceOneOut function
loocv = LeaveOneOut()
r_index = list(r_dataframe.index)
#define filter method function
def filter_method(data,r_index):
    n = data.shape[0]
    p = data.shape[1]-1
    k = 7
    accuracy = []
    for m in range(1,p+1):
        X = data.loc[:,r_index[:m]]
        y = data.iloc[:,p]
        right = 0
        for train_index, test_index in loocv.split(data):
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if KNN_predict(pd.concat([X_train,y_train],axis=1),X_test,k)[0] == y_test.values:
                right += 1
        print(right/n)
        accuracy.append(right/n)
    m = accuracy.index(max(accuracy)) + 1
    acc = accuracy[m-1]
    return m,acc
#print value of m gives the highest LOOCV classiﬁcation accuracy and 
#the value of this optimal accuracy
print(filter_method(data,r_index))
#3. Wrapper Method
def wrapper_method(data,r_index):
    n = data.shape[0]
    p = data.shape[1]-1
    k = 7
    feature_set = []
    print(feature_set)
    feature_set_remain = r_index
    acc = 0
    forward = True
    while forward:
        accuracy = []
        for feature in feature_set_remain:
            temp = feature_set.copy()
            temp.append(feature)
            X = data.loc[:,temp]
            y = data.iloc[:,p]
            right = 0
            for train_index, test_index in loocv.split(data):
                X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                if KNN_predict(pd.concat([X_train,y_train],axis=1),X_test,k)[0] == y_test.values:
                    right += 1
            accuracy.append(right/n)
        if max(accuracy)>acc:
            acc = max(accuracy)
            m = accuracy.index(max(accuracy))
            feature_set.append(feature_set_remain[m])
            print(feature_set)
            del feature_set_remain[m]
        else :
            forward = False 
    return feature_set,acc
#Show the set of selected features at each step and print the 
#LOOCV accuracy over the ﬁnal set of selected features
print(wrapper_method(data,r_index))

