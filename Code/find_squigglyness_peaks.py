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
    mul = 0.15*distance[len(distance)-1]
    segments = [0]
    while(index < len(distance)):
        if(distance[index] > mul):
            segments.append(index)
            mul += 0.15*distance[len(distance)-1]
        index += 1


    plt.scatter(np.arange(0,len(segments)),[y_coordinates[i] for i in segments])
    plt.savefig("woodruff/straight_segments/"+name)
    plt.close()
    
    peaks_freq = []
    peaks_x_coord = []
    peaks = []
    for j in range(1,len(segments)):
        freq = 0
        for k in range(segments[j-1]+1,segments[j]-1):
            if(y_coordinates[k-1] < y_coordinates[k] > y_coordinates[k+1]):
                freq += 1
                peaks.append(y_coordinates[k])
                peaks_x_coord.append(x_coordinates[k])
        peaks_freq.append(freq)
        
    plt.plot(peaks_x_coord,peaks)
    plt.savefig("woodruff/peaks_straight/"+name)
    plt.close()
                
    plt.plot(peaks_freq)
    plt.savefig("woodruff/freq_straight/"+name)
    plt.close()
    try:
        print("activity:",name,"=",sum(peaks_freq)/len(peaks_freq))
    except:
        continue
