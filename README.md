# Terrain_Estimation
Estimating type of surface using GPS data
 
 We have explored how the path taken by a bicycle rider can be sufficient for mapping the type of surface of the path he took. 
 This can be done by using simple mathematics in an inexpensive way unlike the image processing techniques applied onto the 
 satellite images.

Our main aim is to estimate if the road taken by the user is squiggly or straight.In order to do this, two methods are adopted. The first method measures the changes in frequency of direction of slope in a given path segment. A straight path has lower change in frequency of direction of slope than a squiggly path. The second method involves fitting segments of the path using a linear regression model. A squiggly path will have more root mean squared error than a straight path. Machine learning models such as Support vector machine, K nearest neighbors and decision trees is used for the classification of the path into squiggly or straight. Between the two methods and the various classification techniques, decision trees performed the best with an accuracy of 86% .  
