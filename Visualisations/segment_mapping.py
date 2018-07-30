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

ct = 0
print("squiggly")
overall = []
for i in os.listdir("squiggly"):
    name = i
    print("*************")
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
    mul = 50
    segments = [0]
    while(index < len(distance)):
        if(distance[index] > mul):
            segments.append(index)
            mul += 100
        index += 1

    
    peaks_freq = []
    type_segment = []
    overall = []
    for j in range(1,len(segments)):
        freq = 0
        x = [x_coordinates[k] for k in range(segments[j-1],segments[j])]
        y = [y_coordinates[k] for k in range(segments[j-1],segments[j])]
        for k in range(segments[j-1]+1,segments[j]-1):
            if(y_coordinates[k-1] < y_coordinates[k] > y_coordinates[k+1]):
                freq += 1
        if(freq == 0):
            type_segment = "straight"
            plt.plot(x,y,color='blue')
        elif(freq/(distance[segments[j]] - distance[segments[j-1]]) <  0.01):
            type_segment = "straight"
            plt.plot(x,y,color='blue')
        else:
            type_segment = "squiggly"
            plt.plot(x,y,color='red')
        overall.append(freq/(distance[segments[j]] - distance[segments[j-1]]))   
        peaks_freq.append(freq)    
    plt.savefig("map_path_squiggly/"+name)
    plt.close()
    print(overall)
    #print("activity:",name,"=",sum(peaks_freq)/len(peaks_freq))

