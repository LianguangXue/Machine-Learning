# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:43:13 2019

@author: Lianguang Xue
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 

#preprecessing the data
raw_train_1000 = pd.read_csv("train-1000-100.csv")
#create trainning file train-50(1000)-100.csv
raw_train_50_1000 = raw_train_1000.iloc[:50,:]
raw_train_50_1000.to_csv("train-50(1000)-100.csv")
#create trainning file train-50(1000)-100.csv
raw_train_100_1000 = raw_train_1000.iloc[:100,:]
raw_train_100_1000.to_csv("train-100(1000)-100.csv")
#create trainning file train-50(1000)-100.csv
raw_train_150_1000 = raw_train_1000.iloc[:150,:]
raw_train_150_1000.to_csv("train-150(1000)-100.csv")


##2. Regularization 
#load the training and test data
raw_train_100_10 = pd.read_csv("train-100-10.csv")
raw_train_100_100 = pd.read_csv("train-100-100.csv")
raw_test_100_10 = pd.read_csv("test-100-10.csv")
raw_test_100_100 = pd.read_csv("test-100-100.csv")
raw_test_1000 = pd.read_csv("test-1000-100.csv")
#(a).
#Define MSE function to compute training and test data MSE   
def MSE(train_data,test_data,lamda):
    #get the shape of training set
    n_train,p_train = train_data.shape 
    p_train = p_train-1
    #built the matirx for further use 
    X_train = train_data.values[:,:p_train]
    y_train = train_data.values[:,p_train]
    #Implement L2 regularized linear regression algorithm with lamda 
    w = np.linalg.inv(X_train.T@X_train + lamda*np.eye(p_train,k=0))@X_train.T@y_train 
    #compute the training set MSE
    Mse_train = np.sum(np.square(X_train@w - y_train))/n_train
    #get the shape of test set
    n_test,p_test = test_data.shape
    p_test = p_test-1
    X_test = test_data.values[:,:p_test]
    y_test = test_data.values[:,p_test]
    #compute the test set MSE
    Mse_test = np.sum(np.square(X_test@w - y_test))/n_test
    return Mse_train,Mse_test
#Define the plot the function to plot both the training set MSE and the test set MSE as a function of lamda (x-axis) in one graph.
def MSE_plot(train_data,test_data,lamda_range):
    n,p = train_data.shape
    p = p-1
    #get the list of MSE  
    Mse_train = []
    Mse_test = []
    for lamda in range(lamda_range[0],lamda_range[1]+1):
        Mse1,Mse2 = MSE(train_data,test_data,lamda)
        Mse_train.append(Mse1)
        Mse_test.append(Mse2)
    #plot the two MSE in one graph
    plt.figure()
    plt.plot(Mse_train)
    plt.plot(Mse_test)
    plt.title("MSE plot("+str(n)+","+str(p)+")depend on lamda")
    plt.xlabel("Lamda")
    plt.ylabel("MSE")
    plt.show()
    #get the index of the least test set MSE
    ind = Mse_test.index(min(Mse_test))
    #get the value of the lamda
    lamda_value = lamda_range[0]+ind
    #get correspond value of MSE for test data
    MSE_value = MSE(train_data,test_data,lamda_value)[1]
    return lamda_value,MSE_value 
#plot the dataset "train-100-10.csv" MSE plot(lamda ranging from 0 to 150) and print out the lamda value gives the least test set MSE
print(MSE_plot(raw_train_100_10,raw_test_100_10,[0,150]))
#plot the dataset "train-100-100.csv" MSE plot(lamda ranging from 0 to 150) and print out the lamda value gives the least test set MSE
print(MSE_plot(raw_train_100_100,raw_test_100_100,[0,150]))
#plot the dataset "train-50(1000)-100.csv" MSE plot(lamda ranging from 0 to 150) and print out the lamda value gives the least test set MSE
print(MSE_plot(raw_train_50_1000,raw_test_1000,[0,150]))
#plot the dataset "train-100(1000)-100.csv" MSE plot(lamda ranging from 0 to 150) and print out the lamda value gives the least test set MSE
print(MSE_plot(raw_train_100_1000,raw_test_1000,[0,150]))
#plot the dataset "train-150(1000)-100.csv" MSE plot(lamda ranging from 0 to 150) and print out the lamda value gives the least test set MSE
print(MSE_plot(raw_train_150_1000,raw_test_1000,[0,150]))
#plot the dataset "train-1000-100.csv" MSE plot(lamda ranging from 0 to 150) and print out the lamda value gives the least test set MSE
print(MSE_plot(raw_train_1000,raw_test_1000,[0,150]))

#(b).
#provide an additional graph with lamda ranging from 1 to 150 for dataset "train-100-100.csv"
MSE_plot(raw_train_100_100,raw_test_100_100,[1,150])
#provide an additional graph with lamda ranging from 1 to 150 for dataset "train-50(1000)-100.csv"
MSE_plot(raw_train_50_1000,raw_test_1000,[1,150])
#provide an additional graph with lamda ranging from 1 to 150 for dataset "train-100(1000)-100.csv"
MSE_plot(raw_train_100_1000,raw_test_1000,[1,150])

#(c).
#Because with lamda=0, the w will be overfit for the model. It will cause w be suitable for the train data but perform bad in the test data. 
#Therefore the test set MSE is abnormally large.



##3.Cross Validation
#Define the cross validation function
def CrossValidation(data,lamda_list):
    #1.Split the data into 10 disjoint folds
    kf = KFold(n_splits=10) # Define the split - into 10 folds 
    kf.get_n_splits(data)
    perform=[]
    #2.for each value of lamda compute the average performance
    for lamda in lamda_list:
        score = []
        for a,b in kf.split(data):
            #Train on all folds but i^th fold
            MSE_train,MSE_test = MSE(data.iloc[a,:],data.iloc[b,:],lamda)
            score.append(MSE_test)
        #compute the average performance
        perform_lamda = np.mean(score)
        perform.append(perform_lamda)
    #Pick the value of lamda with the best average performance
    lamda_best = lamda_list[perform.index(min(perform))]
    return lamda_best    
#Define the list od the lamda 
ran = list(range(0,151))

#(a).
#print out best choice of lamda and corresponding test set MSE for dataset "train-100-10.csv" 
lamda_100_10 = CrossValidation(raw_train_100_10,ran)
print(lamda_100_10)
print(MSE(raw_train_100_10,raw_test_100_10,lamda_100_10)[1])
#print out best choice of lamda and corresponding test set MSE for dataset "train-100-100.csv" 
lamda_100_100 = CrossValidation(raw_train_100_100,ran)
print(lamda_100_100)
print(MSE(raw_train_100_100,raw_test_100_100,lamda_100_100)[1])
#print out best choice of lamda and corresponding test set MSE for dataset "train-50(1000)-100.csv" 
lamda_50_1000 = CrossValidation(raw_train_50_1000,ran)
print(lamda_50_1000)
print(MSE(raw_train_50_1000,raw_test_1000,lamda_50_1000)[1])
#print out best choice of lamda and corresponding test set MSE for dataset "train-100(1000)-100.csv" 
lamda_100_1000 = CrossValidation(raw_train_100_1000,ran)
print(lamda_100_1000)
print(MSE(raw_train_100_1000,raw_test_1000,lamda_100_1000)[1])
#print out best choice of lamda and corresponding test set MSE for dataset "train-150(1000)-100.csv" 
lamda_150_1000 = CrossValidation(raw_train_150_1000,ran)
print(lamda_150_1000)
print(MSE(raw_train_150_1000,raw_test_1000,lamda_150_1000)[1])
#print out best choice of lamda and corresponding test set MSE for dataset "train-1000-100.csv" 
lamda_1000 = CrossValidation(raw_train_1000,ran)
print(lamda_1000)
print(MSE(raw_train_1000,raw_test_1000,lamda_1000)[1])

#(b).
#From the result of question 2(a) and question 3(a) we get the lamda and MSE
question2_lamda = [8,23,11,25,31,46] 
question2_MSE = [6.443235780122246,6.994249154792671,8.179512636874382,7.74099979657299,7.083190722347952,6.415999409096905]
CV_lamda = [9,12,13,14,37,59]
CV_MSE = [6.444753816207458,7.192220544783701,8.187724827242048,7.8658738365004695,7.093097462024729,6.4218164880281545]
#print ou the results and compare them
print(question2_lamda)
print(CV_lamda)
print(question2_MSE)
print(CV_MSE) 
#From the print out ,we can find that values for lamda and MSE obtained from CV is a little different from the question2 result. 
#However, they are very close and MSE values are also not very large.

#(c).
#The disadvantage of this method is that the training algorithm has to be rerun from scratch k times, 
#which means it takes k times as much computation to make an evaluation. A variant of this method is to 
#randomly divide the data into a test and training set k different times

#(d).
#the list of lamda and choice of K affect the performance of CV.
#The data set we use also affects the performance of CV(like the number of the data)


##4.Learning Curve
#define the learing curve function
def LearningCurve(train_data,test_data,lamda,repeat_time,subsets_num):
    n,p = train_data.shape
    subset = np.linspace(0,n,subsets_num+1)
    i = 0
    score = np.zeros([repeat_time,subsets_num])
    while i < repeat_time:
        for j in range(subsets_num):
            num = np.random.randint(0,n,int(subset[j+1]))
            score[i,j] = MSE(train_data.iloc[num,:],test_data,lamda)[1]
        i += 1
    score_final = np.mean(score, axis=0)
    return score_final
#print out the learing curve list for lamda = 1 (repeat 100 times and 10 subsets(increasingly))
print(LearningCurve(raw_train_1000,raw_test_1000,1,100,10))
plt.figure()
plt.plot(np.linspace(0,1000,11)[1:],LearningCurve(raw_train_1000,raw_test_1000,1,100,10))
plt.title("Learning curve for lamda = 1")
plt.xlabel("Number of Data Points")
plt.ylabel("MSE for Test data")
#print out the learing curve list for lamda = 25 (repeat 100 times and 10 subsets(increasingly))
print(LearningCurve(raw_train_1000,raw_test_1000,25,100,10))
plt.figure()
plt.plot(np.linspace(0,1000,11)[1:],LearningCurve(raw_train_1000,raw_test_1000,25,100,10))
plt.title("Learning curve for lamda = 25")
plt.xlabel("Number of Data Points")
plt.ylabel("MSE for Test data")
#print out the learing curve list for lamda = 150 (repeat 100 times and 10 subsets(increasingly))
print(LearningCurve(raw_train_1000,raw_test_1000,150,100,10))
plt.figure()
plt.plot(np.linspace(0,1000,11)[1:],LearningCurve(raw_train_1000,raw_test_1000,150,100,10))
plt.title("Learning curve for lamda = 150")
plt.xlabel("Number of Data Points")
plt.ylabel("MSE for Test data")
