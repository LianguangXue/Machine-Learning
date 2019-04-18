# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:32:03 2019

@author: Lianguang Xue
"""
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#Question 3
raw_data = arff.loadarff('segment.arff')
data = pd.DataFrame(raw_data[0])
data.head() # look the head of the data
data.shape # look the shape of the data

## preprocessing step
scaler = StandardScaler()  #Normalize data (zero mean, unit variance data)
data_norm = scaler.fit_transform(data.iloc[:,:19])  #Fit to data, then transform it.
data_norm = pd.DataFrame(data_norm,columns=data.iloc[:,:19].columns) #get normalize data
# input the  starting positions of the cluster centroids
instance_list = [775, 1020, 200, 127, 329, 1626, 1515, 651, 658, 328, 1160, 108, 422, 88, 105,\
                261, 212, 1941, 1724, 704, 1469, 635, 867, 1187, 445, 222, 1283, 1288, 1766, \
                1168, 566, 1812, 214, 53, 423, 50, 705, 1284, 1356, 996, 1084, 1956, 254, 711, \
                1997, 1378, 827, 1875, 424, 1790, 633, 208, 1670, 1517, 1902, 1476, 1716, 1709, \
                264, 1, 371, 758, 332, 542, 672, 483, 65, 92, 400, 1079, 1281, 145, 1410, 664, \
                155, 166, 1900, 1134, 1462, 954, 1818, 1679, 832, 1627, 1760, 1330, 913, 234, \
                1635, 1078, 640, 833, 392, 1425, 610, 1353, 1772, 908, 1964, 1260, 784, 520, \
                1363, 544, 426, 1146, 987, 612, 1685, 1121, 1740, 287, 1383, 1923, 1665, 19, 1239, \
                251, 309, 245, 384, 1306, 786, 1814, 7, 1203, 1068, 1493, 859, 233, 1846, 1119, 469, \
                1869, 609, 385, 1182, 1949, 1622, 719, 643, 1692, 1389, 120, 1034, 805, 266, 339, 826, \
                530, 1173, 802, 1495, 504, 1241, 427, 1555, 1597, 692, 178, 774, 1623, 1641, 661, 1242, \
                1757, 553, 1377, 1419, 306, 1838, 211, 356, 541, 1455, 741, 583, 1464, 209, 1615, 475, \
                1903, 555, 1046, 379, 1938, 417, 1747, 342, 1148, 1697, 1785, 298, 1485, 945, 1097, 207, \
                857, 1758, 1390, 172, 587, 455, 1690, 1277, 345, 1166, 1367, 1858, 1427, 1434, 953, 1992, \
                1140, 137, 64, 1448, 991, 1312, 1628, 167, 1042, 1887, 1825, 249, 240, 524, 1098, 311, 337, \
                220, 1913, 727, 1659, 1321, 130, 1904, 561, 1270, 1250, 613, 152, 1440, 473, 1834, 1387, \
                1656, 1028, 1106, 829, 1591, 1699, 1674, 947, 77, 468, 997, 611, 1776, 123, 979, 1471, 1300, \
                1007, 1443, 164, 1881, 1935, 280, 442, 1588, 1033, 79, 1686, 854, 257, 1460, 1380, 495, 1701, \
                1611, 804, 1609, 975, 1181, 582, 816, 1770, 663, 737, 1810, 523, 1243, 944, 1959, 78, 675, 135, \
                1381, 1472]
#define the the fucntion to get start points in the k-means
def start_points(k,instance_list,data,times_num):
    ins_num = instance_list[k*(times_num-1):k*times_num]
    return data.iloc[ins_num,:]
#define the k-means function
def k_means(k, start, data):
    #set initial centriod
    centroid = start.values
    n,p = data.shape
    #set initial cluster
    C = np.repeat(1,n)
    signal = True
    iter_num = 1
    while signal and iter_num <= 50:
        #get distance matrix
        dis_matrix = np.zeros([n,k])
        for i in range(k):
            dis_matrix[:,i] = np.sum((data-centroid[i,:])**2,axis = 1)
        #updata cluster information
        temp = np.argmin(dis_matrix,axis=1)+1
        print(temp)
        #updata the centriod
        for i in range(k):
            centroid[i,:] = np.mean(data[temp==i+1])
        if (temp==C).all():
            signal = False
        else:
            C = temp
            iter_num += 1
    SSE = 0
    for i in range(n):
        SSE += dis_matrix[i,C[i]-1]
    return C,SSE
#get the SSE matrix 
SSE_MAT = np.zeros([25,12])
for times_num in range(1,26):
    for k in range(1,13):
        SSE_MAT[times_num-1,k-1] = k_means(k,start_points(k,instance_list,data_norm,times_num),data_norm)[1]


##(a). For each k = 1,2,...,12 compute the mean SSE, which we denote µk and the sample standard deviation of SSE,
#      which we denote σk, over all 25 clustering runs for that value of k. Generate a line plot of the mean SSE (µk) as a 
#      function of k. Include error bars that indicate the 95% conﬁdence interval: (µk −2σk to µk + 2σk). 


#get mean of SSE for each k
miu = np.mean(SSE_MAT,axis=0) 
print(miu)
#get sample standard deviation of SSE for each k
sigma = np.std(SSE_MAT,axis=0)
print(sigma)
yerr = 2*sigma
#draw the error bar plot
line,caps,bars=plt.errorbar(
    np.arange(1,13,1),     # X
    miu,    # Y
    yerr=yerr,        # Y-errors
    fmt="rs--",    # format line like for plot()
    linewidth=3,   # width of plot line
    elinewidth=0.5,# width of error bar line
    ecolor='k',    # color of error bar
    capsize=5,     # cap length for error bar
    capthick=0.5   # cap thickness for error bar
    )

plt.setp(line,label="My error bars")#give label to returned line
plt.legend(numpoints=1,             #Set the number of markers in label
           loc=('upper right'))      #Set label location
plt.xlim((0,13))                 #Set X-axis limits
plt.show()

## (b). 
#  Produce a table containing the 4 columns: k, µk, µk −2σk and µk + 2σk for each of the values of 
#  k = 1,2,...,12. 
table = pd.DataFrame(miu,index = ["k=1","k=2","k=3","k=4","k=5","k=6","k=7","k=8","k=9","k=10","k=11","k=12"],
                     columns = ["miu"])
table["sigma"] = sigma
table["miu-2*sigma"] = miu-2*sigma
table["miu+2*sigma"] = miu+2*sigma
print(table)

#Question 5
c1 = np.array([[1,1],[2,2],[3,3]])
c2 = np.array([[5,2],[6,2],[7,2],[8,2],[9,2]])
#(a).The mean vectors m1 and m2 
m1 = np.mean(c1,axis = 0)
print(m1)
m2 = np.mean(c2,axis = 0)
print(m2)
#(b).The total mean vector m 
m = np.mean(np.concatenate((c1,c2),axis=0),axis=0)
print(m)
#(c).The scatter matrices S1 and S2 
s1 = (c1-m1).T@(c1-m1)
print(s1)
s2 = (c2-m2).T@(c2-m2)
print(s2)
#(d).The within-cluster scatter matrix SW 
sw =s1 + s2
print(sw)
#(e).The between-cluster scatter matrix SB 
sb = 3*(c1-m).T@(c1-m)+5*(c2-m).T@(c2-m)
print(sb)
#(f). The scatter criterion tr(SB)/tr(SW )
print(np.trace(sb)/np.trace(sw))







