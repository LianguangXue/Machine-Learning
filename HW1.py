# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 23:00:00 2019

@author: Lianguang Xue
"""
from scipy.io import arff
import pandas as pd
import numpy as np
from collections import Counter

raw_data_train = arff.loadarff('train.arff')
raw_data_test = arff.loadarff('test.arff')

data_train = pd.DataFrame(raw_data_train[0])
data_test = pd.DataFrame(raw_data_test[0])

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
    
def KNN_predict(data_train,data_test,k):  
    classifer = {}
    class_predict = []
    for i in range(0,len(data_test)):
        temp = []  
        for j in range(0,len(data_train)):
#            print(data_test[i:(i+1)])
#            print(data_test[i:(i+1)]-data_train.iloc[j:(j+1),0:4])
            diff = np.array(data_test[i:i+1])-np.array(data_train.iloc[j:j+1,0:4])
#            print(diff)
#            print(sum([c*c for c in diff[0]]))
            temp.append(sum([c*c for c in diff[0]]))
#            print(temp)
        sort_temp = sorted(enumerate(temp), key=lambda x:x[1])
        index = [x[0] for x in sort_temp]
        index_distance = [x[1] for x in sort_temp][0:k]
#        print(index)
        class_label = []
        for a in index[0:k]:
            class_label.append(np.array(data_train.iloc[a:(a+1),4:5]).tolist()[0][0])
        classifer[i] = class_label
#    print(classifer)
    for i in range(0,len(data_test)):
        class_predict.append(vote(classifer[i],index_distance))
#    print(class_predict)
    return class_predict
col_name = list(data_test.columns) + ['CLASS_LABEL k = 1','CLASS_LABEL k = 3',\
                               'CLASS_LABEL k = 5','CLASS_LABEL k = 7','CLASS_LABEL k = 9'] 

predict_1 = KNN_predict(data_train,data_test,1)
predict_3 = KNN_predict(data_train,data_test,3)
predict_5 = KNN_predict(data_train,data_test,5)
predict_7 = KNN_predict(data_train,data_test,7)
predict_9 = KNN_predict(data_train,data_test,9)
data_test_predict = data_test.reindex(columns=col_name,fill_value = 0)
data_test_predict['CLASS_LABEL k = 1'] = predict_1
data_test_predict['CLASS_LABEL k = 3'] = predict_3
data_test_predict['CLASS_LABEL k = 5'] = predict_5
data_test_predict['CLASS_LABEL k = 7'] = predict_7
data_test_predict['CLASS_LABEL k = 9'] = predict_9
data_test_predict
#print(data_train[0:1]) 
#print(data_test)
