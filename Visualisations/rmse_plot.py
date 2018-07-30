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
    print("squiggly")
    for i in os.listdir("nitish/squiggly"):
        name = i
        data = pd.read_pickle("nitish/squiggly/"+i)
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
        rms = []
        rmse_distance = []

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
		       
            rmse_distance.append(seg)
            rms.append(sqrt(sum(error)/len(error)))
            seg +=  0.01*distance[len(distance)-1]
            
        plt.hist(rms)
        plt.title("variation of RMSE")
        plt.xlabel("Distance")
        plt.ylabel("RMSE")
        plt.savefig("nitish/rms_squiggly/"+i)
        plt.close()     

        
main()
