import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
import statistics
import pytz
from datetime import datetime,timedelta
import calendar
import time as time_module
import requests
import pickle

model = pickle.load(open("SVM_model",'rb'))
overall = []
overall_peaks = []
ct = 0

with open('openuv_key.json', 'r') as key:
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
    print("nitish")
    global ct
    for i in os.listdir("nitish_all_activities/"):
        name = i
        data = pd.read_pickle("nitish_all_activities/"+i)
        try:
            latlng = data["latlng"]
            altitude = list(data["altitude"])
            distance = list(data["distance"])
            time = list(data["time"])
            timezone = data["timezone"]
            start_date_local = data["start_date_local"]
        except KeyError:
	        continue
	        
        x_coordinates,y_coordinates = obtain_xycoordinates(latlng,altitude) 
	    	    
        segments = obtain_segments(distance)
        
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
        

        if(len(peaks_freq) == 100):
           del(peaks_freq[-1])
        if(len(peaks_freq)==99):
           overall_peaks.append(peaks_freq)
        ct += 1
	    
main()
for j in overall_peaks:
	if(len(j)!=99):
		print(len(j))
predicted = model.predict(overall_peaks)
print(predicted)
result = np.bincount(predicted)
print(result)
values = [result[0],result[1]]
labels = ["Squiggly","Straight"]
pie = plt.pie(values,labels = labels, autopct='%1.1f%%',radius = 0.7 ,shadow=False)
plt.legend(pie[0],labels,loc='lower right')
plt.title("Distribution of Rides for Athlete Id: 145279")
plt.show()
plt.savefig("dist_rides")

