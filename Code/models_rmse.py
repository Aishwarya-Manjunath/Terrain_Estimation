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
    
    rms = []
    model = linear_model.LinearRegression()
    for j in range(1,len(segments)):
        if(segments[j] - segments[j-1] < 8):
           continue
        else:
           x_seg = np.array([x_coordinates[k] for k in range(segments[j-1],segments[j])])
           y_seg = np.array([y_coordinates[k] for k in range(segments[j-1],segments[j])])
           X_train, X_test, y_train, y_test = train_test_split(x_seg, y_seg, test_size=0.50)
           model.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))
           y_predicted = model.predict(X_test.reshape(-1,1))
           error = [(X_test[k] - y_predicted[k])**2 for k in range(0,len(y_predicted))]
           rms.append(sqrt(sum(error)/len(error)))
     
        
    if(len(rms) == 100):
        del(rms[-1])
    '''
    print(len(peaks_freq))
    
    try:
        print("activity:",name,"=",sum(peaks_freq)/len(peaks_freq))
    except:
        continue
    '''
    if(len(rms) == 99):
        '''
        count = 0
        for j in range(1,len(rms)-1):
            if(rms[j-1] < rms[j] > rms[j+1]):
               count += 1
        '''
        overall_straight.append(rms)
    
    
    
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
    
    rms = []
    model = linear_model.LinearRegression()
    for j in range(1,len(segments)):
        if(segments[j] - segments[j-1] < 8):
           continue
        else:
           x_seg = np.array([x_coordinates[k] for k in range(segments[j-1],segments[j])])
           y_seg = np.array([y_coordinates[k] for k in range(segments[j-1],segments[j])])
           X_train, X_test, y_train, y_test = train_test_split(x_seg, y_seg, test_size=0.50)
           model.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))
           y_predicted = model.predict(X_test.reshape(-1,1))
           error = [(X_test[k] - y_predicted[k])**2 for k in range(0,len(y_predicted))]
           rms.append(sqrt(sum(error)/len(error)))
        

    if(len(rms) == 100):
        del(rms[-1])
    '''
    print(len(peaks_freq))
    
    try:
        print("activity:",name,"=",sum(peaks_freq)/len(peaks_freq))
    except:
        continue
    ''' 
    if(len(rms) == 99):
        '''
        count = 0
        for j in range(1,len(rms)-1):
            if(rms[j-1] < rms[j] > rms[j+1]):
               count += 1
        '''
        overall_squiggly.append(rms)
    
  
#straight: 1
#squiggly: 0
# 50% train and 50% test

seed = 999699

x_train_st ,x_test_st ,y_train_st ,y_test_st = train_test_split(overall_straight,[1]*len(overall_straight),test_size=0.3,random_state = seed, stratify = [1]*len(overall_straight))


x_train_sq ,x_test_sq ,y_train_sq ,y_test_sq = train_test_split(overall_squiggly,[0]*len(overall_squiggly),test_size=0.3,random_state = seed, stratify = [0]*len(overall_squiggly))

x_train = np.array(x_train_st + x_train_sq)
x_test = np.array(x_test_st + x_test_sq)
y_train = np.array(y_train_st + y_train_sq)
y_test = np.array(y_test_st + y_test_sq)


clf = svm.SVC()
clf.fit(x_train,y_train)
predicted = clf.predict(x_test)
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

neigh = KNeighborsClassifier(n_neighbors = 3)
neigh.fit(x_train,y_train)
predicted = neigh.predict(x_test)
print("KNN Accuracy: ",accuracy_score(y_test,predicted)*100)
print(classification_report(y_test,predicted))

print(classification_report(y_test,predicted))
# Compute ROC curve and ROC area 
fpr, tpr, _ = roc_curve(y_test, predicted)
roc_auc = auc(fpr, tpr)
#plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')


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


mlp = MLPClassifier(activation='relu',learning_rate='adaptive',learning_rate_init=0.1)
mlp.fit(x_train,y_train)
predicted = mlp.predict(x_test)
print("Multilayer perceptron Accuracy: ",accuracy_score(y_test,predicted)*100)
print(classification_report(y_test,predicted))
