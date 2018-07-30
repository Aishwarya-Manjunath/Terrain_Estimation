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
import pytz
from datetime import datetime,timedelta
import calendar
import time
import requests

with open('darksky_key.json', 'r') as key:
    key = json.load(key)

key = key["key"]
    
    
def obtain_xycoordinates(latlng,altitude):
    x_coordinates = []
    y_coordinates = []
    radius_earth = 6371000
    index = 0
    for i in latlng:
        height = radius_earth + altitude[index]
        index += 1
        x_coordinates.append(height*math.cos(math.radians(i[0]))*math.cos(math.radians(i[1])))
        y_coordinates.append(height*math.cos(math.radians(i[0]))*math.sin(math.radians(i[1])))
       
    return x_coordinates,y_coordinates


def obtain_segments(distance):
    index = 0
    mul = 0.01*distance[len(distance)-1]
    segments = [0]
    while(index < len(distance)):
        if(distance[index] > mul):
            segments.append(index)
            mul += 0.01*distance[len(distance)-1]
        index += 1
    
    return segments
    
       

def main():
    print("straight")
    for i in os.listdir("nitish/straight"):
        name = i
        data = pd.read_pickle("nitish/straight/"+i)
        try:
            latlng = data["latlng"]
            altitude = list(data["altitude"])
            distance = list(data["distance"])
            time = list(data["time"])
            timezone = data["timezone"]
            start_date_local = data["start_date_local"]
        except KeyError:
	        continue
	        
        seg =  0.01*distance[len(distance)-1]    
        x_coordinates,y_coordinates = obtain_xycoordinates(latlng,altitude) 
	    	    
        segments = obtain_segments(distance)
        
        peaks_freq = []
        peaks_x_coord = []
        peaks = []
        peaks_distance = []
        for j in range(1,len(segments)):
            freq = 0
            for k in range(segments[j-1]+1,segments[j]-1):
                if(y_coordinates[k-1] < y_coordinates[k] > y_coordinates[k+1]):
                    freq += 1
                    peaks.append(y_coordinates[k])
                    peaks_x_coord.append(x_coordinates[k])
            peaks_freq.append(freq)
            peaks_distance.append(seg)
            seg +=  0.01*distance[len(distance)-1]
            
        plt.plot(peaks_distance,peaks_freq)
        plt.title("variation of frequencies")
        plt.xlabel("Distance")
        plt.ylabel("Frequency of change in direction of slope")
        plt.savefig("nitish/freq_straight/"+i)
        plt.close()     

        
main()
