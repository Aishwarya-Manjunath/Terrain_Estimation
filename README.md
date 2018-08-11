# Terrain_Estimation
Estimating type of surface using GPS data
 
 We have explored how the GPS coordinates of the path taken by a bicycle rider can be sufficient for mapping the type of surface of the path he took. 
 This can be done by using simple mathematics in an inexpensive way unlike the image processing techniques applied onto the 
 satellite images.

Our main aim is to estimate if the road taken by the user is squiggly or straight.In order to do this, two methods are adopted. The first method measures the changes in frequency of direction of slope in a given path segment. A straight path has lower change in frequency of direction of slope than a squiggly path. The second method involves fitting segments of the path using a linear regression model. A squiggly path will have more root mean squared error than a straight path. Machine learning models can be used for the classification of the path into squiggly or straight. 

<b>Directory Structure:</b>


1. Code:
    
    find_squigglyness_peaks.py - has the code to find the squigglyness w.r.t to frequency of change in direction of slopes.
    
    find_squigglyness_rmse.py - has the code to find squigglyness based on root mean square error obtained by fitting the segmented
    path using Linear regression model.
    
    models_peaks.py - has the code to build SVM, KNN, Decision trees using frequency of change in direction of slopes.
    
    models_rmse.py - has the code to build SVM, KNN, Decision trees using rmse.
 
    models_zc.py - has the code to build SVM, KNN, Decision trees using zero crossings of first derivative.
    
    distribution_user_rides.py - has code to find percenntage of squiggly and straight paths taken by the rider.
    
    
 2. Visualisations:
 
     frequency_plot.py - has code to plot variation in frequency in segmented paths.
     
     rmse_plot.py - has the code to plot histogram of the rmse.

     zc_plot.py - has the code to plot number of points of zero crossings in a segment.
     
     segment_mapping.py - it contains code to find if each segment of the road is squiggly or straight and plot the same.
     
     
