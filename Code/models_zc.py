import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peakutils
from math import sqrt
from itertools import cycle
from scipy import interp

from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

#15

ct = 0
#print("straight")
overall_straight = []
for i in os.listdir("straight"):
    name = i
    data = pd.read_pickle("straight/"+i)
    try:
        latlng = data["latlng"]
        altitude = list(data["altitude"])
        distance = list(data["distance"])
    except KeyError:
	    continue
    ct += 1
    x_coordinates = []
    y_coordinates = []
    radius_earth = 6371000
    index = 0
    for i in latlng:
        height = radius_earth + altitude[index]
        index += 1
        x_coordinates.append(height*math.cos(math.radians(i[0]))*math.cos(math.radians(i[1])))
        y_coordinates.append(height*math.cos(math.radians(i[0]))*math.sin(math.radians(i[1])))

    index = 0
    mul = 0.01*distance[len(distance)-1]
    segments = [0]
    while(index < len(distance)):
        if(distance[index] > mul):
            segments.append(index)
            mul += 0.01*distance[len(distance)-1]
        index += 1
    
    zero_crossings = []
    for j in range(1,len(segments)):
        freq = 0
        x = []
        y = []
        for k in range(segments[j-1]+1,segments[j]-1):
            y.append(y_coordinates[k])
            x.append(x_coordinates[k])
        derivative = np.diff(y)/np.diff(x)
        count = 0
        for i in derivative:
            if(i<0.009 and i >-0.009):
              count += 1
        zero_crossings.append(count)
     
        
    if(len(zero_crossings) == 100):
        del(zero_crossings[-1])
    '''
    print(len(peaks_freq))
    
    try:
        print("activity:",name,"=",sum(peaks_freq)/len(peaks_freq))
    except:
        continue
    '''
    if(len(zero_crossings) == 99):
        overall_straight.append(zero_crossings)
    
    
    
ct = 0
#print("squiggly")
overall_squiggly = []
for i in os.listdir("squiggly"):
    name = i
    data = pd.read_pickle("squiggly/"+i)
    try:
        latlng = data["latlng"]
        altitude = list(data["altitude"])
        distance = list(data["distance"])
    except KeyError:
	    continue
    ct += 1
    x_coordinates = []
    y_coordinates = []
    radius_earth = 6371000
    index = 0
    for i in latlng:
        height = radius_earth + altitude[index]
        index += 1
        x_coordinates.append(height*math.cos(math.radians(i[0]))*math.cos(math.radians(i[1])))
        y_coordinates.append(height*math.cos(math.radians(i[0]))*math.sin(math.radians(i[1])))

    index = 0
    mul = 0.01*distance[len(distance)-1]
    segments = [0]
    while(index < len(distance)):
        if(distance[index] > mul):
            segments.append(index)
            mul += 0.01*distance[len(distance)-1]
        index += 1
    
    zero_crossings = []
    for j in range(1,len(segments)):
        freq = 0
        x = []
        y = []
        for k in range(segments[j-1]+1,segments[j]-1):
            y.append(y_coordinates[k])
            x.append(x_coordinates[k])
        derivative = np.diff(y)/np.diff(x)
        count = 0
        for i in derivative:
            if(i<0.009 and i >-0.01):
              count += 1
        zero_crossings.append(count)
        

    if(len(zero_crossings) == 100):
        del(zero_crossings[-1])
    '''
    print(len(peaks_freq))
    
    try:
        print("activity:",name,"=",sum(peaks_freq)/len(peaks_freq))
    except:
        continue
    ''' 
    if(len(zero_crossings) == 99):
        overall_squiggly.append(zero_crossings)
    
  
#print("straight: ",len(overall_straight))
#print("squiggly: ",len(overall_squiggly))

#straight: 1
#squiggly: 0
# 50% train and 50% test

seed = 123

x_train_st ,x_test_st ,y_train_st ,y_test_st = train_test_split(overall_straight,[1]*len(overall_straight),test_size=0.45,random_state = seed, stratify = [1]*len(overall_straight))

#print("x_test_st:  ",y_test_st)

x_train_sq ,x_test_sq ,y_train_sq ,y_test_sq = train_test_split(overall_squiggly,[0]*len(overall_squiggly),test_size=0.45,random_state = seed, stratify = [0]*len(overall_squiggly))

x_train = np.array(x_train_st + x_train_sq)
x_test = np.array(x_test_st + x_test_sq)
y_train = np.array(y_train_st + y_train_sq)
y_test = np.array(y_test_st + y_test_sq)

#print(x_train)
#print(len(x_train),len(y_train),len(x_test),len(y_test))

clf = svm.SVC()
clf.fit(x_train,y_train)
predicted = clf.predict(x_test)
#print("predicted: ",predicted)
#print("y_test: ",y_test)
print("SVM Accuracy: ",accuracy_score(y_test,predicted)*100)
print(classification_report(y_test,predicted))

# Compute ROC curve and ROC area 
fpr, tpr, _ = roc_curve(y_test, predicted)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for various models')

'''
Accuracies :
    20-80: 79.12
    30-70: 77.50
    50-50: 73.68
    60-40: 67.39
    65-35: 65.00
    70-30: 68.57
    75-25: 72.41
    80-20: 70.83
    90-10: 69.23
'''

neigh = KNeighborsClassifier(n_neighbors = 3)
neigh.fit(x_train,y_train)
predicted = neigh.predict(x_test)
print("KNN Accuracy: ",accuracy_score(y_test,predicted)*100)
print(classification_report(y_test,predicted))
# Compute ROC curve and ROC area 
fpr, tpr, _ = roc_curve(y_test, predicted)
roc_auc = auc(fpr, tpr)
#plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
'''
Accuracies:
    75-25 with k = 2: 65.517
    75-25 with k = 3: 72.413
    75-25 with k = 4: 72.413
    75-25 with k = 5: 68.965
'''

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(x_train,y_train)
predicted = clf.predict(x_test)
print("Decision Trees Accuracy: ",accuracy_score(y_test,predicted)*100)
print(classification_report(y_test,predicted))
# Compute ROC curve and ROC area 
fpr, tpr, _ = roc_curve(y_test, predicted)
roc_auc = auc(fpr, tpr)
#plt.figure()
lw = 2
plt.plot(fpr, tpr, color='green',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.legend(loc="lower right")
plt.show()

'''
mlp = MLPClassifier(activation='relu',learning_rate='adaptive',learning_rate_init=0.1)
mlp.fit(x_train,y_train)
predicted = mlp.predict(x_test)
print("Multilayer perceptron Accuracy: ",accuracy_score(y_test,predicted)*100)
print(classification_report(y_test,predicted))
# Compute ROC curve and ROC area 
fpr, tpr, _ = roc_curve(y_test, predicted)
roc_auc = auc(fpr, tpr)
#plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for ANN')
plt.legend(loc="lower right")
plt.show()
'''




'''
###########################################################################################
FOR 75% train and 25% test data with K=3 for KNN and max_depth=3 for DT: seed=999
###########################################################################################

SVM Accuracy:  72.41379310344827
             precision    recall  f1-score   support

          0       0.67      0.77      0.71        13
          1       0.79      0.69      0.73        16

avg / total       0.73      0.72      0.72        29

******************************************************************************************************************

KNN Accuracy:  72.41379310344827
             precision    recall  f1-score   support

          0       1.00      0.38      0.56        13
          1       0.67      1.00      0.80        16

avg / total       0.82      0.72      0.69        29

*******************************************************************************************************************

Decision Trees Accuracy:  68.96551724137932
             precision    recall  f1-score   support

          0       0.67      0.62      0.64        13
          1       0.71      0.75      0.73        16

avg / total       0.69      0.69      0.69        29


###########################################################################################
FOR 75% train and 25% test data with K=3 for KNN and max_depth=3 for DT: seed=123
###########################################################################################
SVM Accuracy:  86.20689655172413
             precision    recall  f1-score   support

          0       0.80      0.92      0.86        13
          1       0.93      0.81      0.87        16

avg / total       0.87      0.86      0.86        29

KNN Accuracy:  72.41379310344827
             precision    recall  f1-score   support

          0       1.00      0.38      0.56        13
          1       0.67      1.00      0.80        16

avg / total       0.82      0.72      0.69        29

Decision Trees Accuracy:  72.41379310344827
             precision    recall  f1-score   support

          0       0.78      0.54      0.64        13
          1       0.70      0.88      0.78        16

avg / total       0.73      0.72      0.71        29

Multilayer perceptron Accuracy:  48.275862068965516
             precision    recall  f1-score   support

          0       0.46      0.92      0.62        13
          1       0.67      0.12      0.21        16

avg / total       0.57      0.48      0.39        29

###########################################################################################
FOR 50% train and 50% test data with K=3 for KNN and max_depth=3 for DT:
###########################################################################################

SVM Accuracy:  73.68421052631578
             precision    recall  f1-score   support

          0       0.68      0.81      0.74        26
          1       0.81      0.68      0.74        31

avg / total       0.75      0.74      0.74        57

******************************************************************************************************************

KNN Accuracy:  68.42105263157895
             precision    recall  f1-score   support

          0       1.00      0.31      0.47        26
          1       0.63      1.00      0.78        31

avg / total       0.80      0.68      0.64        57

******************************************************************************************************************

Decision Trees Accuracy:  71.9298245614035
             precision    recall  f1-score   support

          0       0.75      0.58      0.65        26
          1       0.70      0.84      0.76        31

avg / total       0.72      0.72      0.71        57


'''
