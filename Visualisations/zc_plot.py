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

#15

ct = 0
print("squiggly")
overall = []
for i in os.listdir("nitish/squiggly"):
    name = i
    data = pd.read_pickle("nitish/squiggly/"+i)
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
    plt.savefig("nitish/paths_squiggly/"+name)
    plt.close()
    
    
    index = 0
    mul = 0.15*distance[len(distance)-1]
    segments = [0]
    while(index < len(distance)):
        if(distance[index] > mul):
            segments.append(index)
            mul += 0.15*distance[len(distance)-1]
        index += 1


    plt.scatter(np.arange(0,len(segments)),[y_coordinates[i] for i in segments])
    plt.savefig("nitish/squiggly_segments/"+name)
    plt.close()
    
    zero_crossings = []
    x_zero_crossings = []
    ct = 1
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
        x_zero_crossings.append(ct)
        ct += 1    
    plt.bar(x_zero_crossings,zero_crossings)
    plt.savefig("nitish/zeroCross_squiggly/"+name)
    plt.close()
