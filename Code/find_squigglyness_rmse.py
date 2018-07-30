import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import peakutils
import statistics

ct = 0
print("straight")
overall = []
for i in os.listdir("woodruff/straight"):
    name = i
    data = pd.read_pickle("woodruff/straight/"+i)
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

    
    plt.plot(x_coordinates,y_coordinates)
    plt.savefig("woodruff/paths_straight/"+name)
    plt.close()
    
    
    index = 0
    mul = 25
    segments = [0]
    while(index < len(distance)):
        if(distance[index] > mul):
            segments.append(index)
            mul += 25
        index += 1


    plt.scatter(np.arange(0,len(segments)),[y_coordinates[i] for i in segments])
    plt.savefig("woodruff/straight_segments/"+name)
    plt.close()
    
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

    plt.plot(rms)
    plt.savefig("woodruff/rms_straight/"+str(name))
    plt.close()
    
    ct = 0
    for j in range(1,len(rms)-1):
        if(rms[j-1] < rms[j] > rms[j+1]):
           ct += 1
        
    print("For user:",name,"=",ct)

