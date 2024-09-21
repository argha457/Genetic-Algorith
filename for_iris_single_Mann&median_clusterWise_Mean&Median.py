import pandas as pd # reading all required header files
import numpy as np
import pandas as pd # reading all required header files
import numpy as np
import random
import operator
import math
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal 
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min

##############################################################################
# Load Dataset
##############################################################################

df = pd.read_csv(r'D:\final_year_project_work\final_year_project_work\genetic_algo_with_iris_dataset.py')
#print(df)
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'species' column
df['species'] = label_encoder.fit_transform(df['species'])
#print(target)
df= df.drop(columns=['species']) # Class labels
#print(df)
_,col_num = df.shape

#print(df.columns)

#print(df.head())
##############################################################################
#number of data
n = len(df)
#number of clusters
k = 3
#dimension of cluster
d = col_num
# m parameter
m = 2
#number of iterations
MAX_ITERS = 12
##############################################################################

##############################################################################
# Initializing Membership Matrix
# wij values are assigned randomly.
##############################################################################
def initializeMembershipWeights():
  weight = np.random.dirichlet(np.ones(k),n)
  weight_arr = np.array(weight)
  return weight_arr

##############################################################################
# Calculating Cluster Center
# To calculate centroids for each cluster we apply the following formula:
# Cj and m(fuzzy-ness) ranges from 1 to infinity
##############################################################################

def computeCentroids(weight_arr,df):
  C = []
  for i in range(k):
    weight_sum = np.power(weight_arr[:,i],m).sum()
    Cj = []
    for x in range(d):
      numerator = ( df.iloc[:,x].values * np.power(weight_arr[:,i],m)).sum()
      c_val = numerator/weight_sum;
      Cj.append(c_val)
    C.append(Cj)
  return C 


##############################################################################
# Updating Membership Value
# Calculate the fuzzy-pseudo partition with the above formula
# $$w_{ij} = \frac{(\frac{1}{dist(x_i, c_j)})^{\frac{1}{m-1}}}{\sum_{s=1}^{k}(\frac{1}{dist(x_i,c_s)})^{\frac{1}{m-1}}}w
# ij$$
# 
# where 
#  dist(x_i, c_j) is the Euclidean distance between x_i and c_j cluster center.
##############################################################################


def updateWeights(weight_arr,C,df):
  denom = np.zeros(n)
  for i in range(k):
    dist = (df.iloc[:,:].values - C[i])**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    denom  = denom + np.power(1/dist,1/(m-1))

  for i in range(k):
    dist = (df.iloc[:,:].values - C[i])**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    weight_arr[:,i] = np.divide(np.power(1/dist,1/(m-1)),denom)
  return weight_arr
##############################################################################
# finding the highest membership and next highest (2nd highest) membership value
##############################################################################
def findHighestAndNextHighestElement(m):
    #Declaring arrays like highest, nextHighest, diff
    highest = np.zeros(len(m))
    nextHighest = np.zeros(len(m))
    diff = np.zeros(len(m))
    m = np.sort(m)
    for i in range(len(m)):
        highest[i] = m[i][len(m[i])-1]
        nextHighest[i] = m[i][len(m[i])-2]
        diff[i] = highest[i] - nextHighest[i]
    return (highest, nextHighest, diff)

##############################################################################
# Fuzzy algorithm
#############################################################################
def FuzzyMeansAlgorithm(df):
    weight_arr = initializeMembershipWeights()
    for z in range(MAX_ITERS):
        C = computeCentroids(weight_arr,df)
        updateWeights(weight_arr,C,df)
 
    return (weight_arr,C)

#calling the FuzzyMeansAlgorithm.
final_weights,Centers = FuzzyMeansAlgorithm(df)
#Finding highest and next highest membership value
highest, nextHighest, diff = findHighestAndNextHighestElement(final_weights)
def singleThresholdCalculation(diff):
    singleMeanThreshold = np.sum(diff) / len(diff)
    sortedDiff = np.sort(diff)
    n = len(sortedDiff)
    if len(diff)%2 != 0:
        singleMedianThreshold = sortedDiff[(n+1)/2]
    else:
        singleMedianThreshold = (sortedDiff[int(n/2)] + sortedDiff[int((n/2)+1)]) / 2
    return (singleMeanThreshold, singleMedianThreshold)

######################################################################################
##Spliting weights in to appropriate clusters
######################################################################################
def clusterDataPoints(final_wt, cen, arr):
  cluster1 = np.zeros((1000,col_num), float)
  cluster2 = np.zeros((1000,col_num), float)
  cluster3 = np.zeros((1000,col_num), float)
  cnt1 = 0
  cnt2 = 0
  cnt3 = 0
  for i in range(len(final_wt)):
    if final_wt[i][0] == max(final_wt[i]):
        cluster1[cnt1] = arr[i]
        cnt1 += 1
    elif final_wt[i][1] == max(final_wt[i]):
        cluster2[cnt2] = arr[i]
        cnt2 += 1
    else:
        cluster3[cnt3] = arr[i]
        cnt3 += 1

  return (cluster1, cnt1, cluster2, cnt2, cluster3, cnt3)
    
##############################################################################
# Function to calculate threshold values for different clusters
##############################################################################
def clusterThresholdCalculation(cluster):
    clusterMeanThreshold = np.sum(cluster) / len(cluster)
    sortedCluster = np.sort(cluster)
    n = len(sortedCluster)
    if len(diff)%2 != 0:
        clusterMedianThreshold = sortedCluster[(n+1)/2]
    else:
        clusterMedianThreshold = (sortedCluster[int(n/2)] + sortedCluster[int((n/2)+1)]) / 2
    return (clusterMeanThreshold, clusterMedianThreshold)

##############################################################################
#Function to find core and boundary data points wrt Mean Threshold (both single mean and cluster-wise mean)
##############################################################################
def findingCore_boundary_MeanThreshold(cluster, MeanThreshold):
    #print(MeanThreshold)
    coreMeanCluster = np.zeros((len(cluster),col_num), float)
    boundaryMeanCluster = np.zeros((len(cluster),col_num), float)
    cnt1=0
    cnt2=0
    #Checking if mean value less than singleMean, then it should go to core else it should go to boundary
    for i in range(len(cluster)):
        if (cluster[i]<MeanThreshold).any():
            coreMeanCluster[cnt1] = cluster[i]
            cnt1 += 1
        else:
            boundaryMeanCluster[cnt2] = cluster[i]
            cnt2 += 1
    
    coreMeanCluster = coreMeanCluster[0:cnt1,:]
    boundaryMeanCluster = boundaryMeanCluster[0:cnt2,:]
    
    return coreMeanCluster, boundaryMeanCluster

##############################################################################
#Function to find core and boundary data points wrt Median Threshold (both single median and cluster-wise median)
##############################################################################
def findingCore_boundary_MedianThreshold(cluster, MedianThreshold):
    coreMedianCluster = np.zeros((len(cluster),col_num), float)
    boundaryMedianCluster = np.zeros((len(cluster),col_num), float)
    cnt1=0
    cnt2=0
    #Checking if mean value less than singleMean, then it should go to core else it should go to boundary
    for i in range(len(cluster)):
        #if (np.median(cluster[i]))<MedianThreshold:
        if (cluster[i]<MedianThreshold).any():
            coreMedianCluster[cnt1] = cluster[i]
            cnt1 += 1
        else:
            boundaryMedianCluster[cnt2] = cluster[i]
            cnt2 += 1
    
    coreMedianCluster = coreMedianCluster[0:cnt1,:]
    boundaryMedianCluster = boundaryMedianCluster[0:cnt2,:]
    
    return coreMedianCluster, boundaryMedianCluster
    
  



#############################################################################################
##############################################################################################


##############################################################################
# Running algorithm
##############################################################################
final_weights,Centers = FuzzyMeansAlgorithm(df)


#Finding highest and next highest membership value
highest, nextHighest, diff = findHighestAndNextHighestElement(final_weights)

#SingleMeanThreshold and singleMedianThreshold calculation
singleMeanThreshold, singleMedianThreshold = singleThresholdCalculation(diff)

#Spliting weights in to appropriate clusters
arr = df.to_numpy()
cluster1, cluster1_count, cluster2, cluster2_count, cluster3, cluster3_count = clusterDataPoints(final_weights, Centers, arr)
cluster1 = cluster1[0:cluster1_count,:]
cluster2 = cluster2[0:cluster2_count,:]
cluster3 = cluster3[0:cluster3_count,:]


#Cluster-wise mean and median claculation
cluster1MeanThreshold, cluster1MedianThreshold = clusterThresholdCalculation(cluster1)
cluster2MeanThreshold, cluster2MedianThreshold = clusterThresholdCalculation(cluster2)
cluster3MeanThreshold, cluster3MedianThreshold = clusterThresholdCalculation(cluster3)


#Seperating data points into core and boundary categoris based on singleMeanThreshold value
coreSingleMeanCluster1, boundarySingleMeanCluster1 = findingCore_boundary_MeanThreshold(cluster1, singleMeanThreshold)
coreSingleMeanCluster2, boundarySingleMeanCluster2 = findingCore_boundary_MeanThreshold(cluster2, singleMeanThreshold)
coreSingleMeanCluster3, boundarySingleMeanCluster3 = findingCore_boundary_MeanThreshold(cluster3, singleMeanThreshold)


#Seperating data points into core and boundary categoris based on singleMedianThreshold value
coreSingleMedianCluster1, boundarySingleMedianCluster1 = findingCore_boundary_MedianThreshold(cluster1, singleMedianThreshold)
coreSingleMedianCluster2, boundarySingleMedianCluster2 = findingCore_boundary_MedianThreshold(cluster2, singleMedianThreshold)
coreSingleMedianCluster3, boundarySingleMedianCluster3 = findingCore_boundary_MedianThreshold(cluster3, singleMedianThreshold)


#Seperating data points into core and boundary categoris based on clusterWiseMeanThreshold value
coreClusterwiseMeanCluster1, boundaryClusterwiseMeanCluster1 = findingCore_boundary_MeanThreshold(cluster1, cluster1MeanThreshold)
coreClusterwiseMeanCluster2, boundaryClusterwiseMeanCluster2 = findingCore_boundary_MeanThreshold(cluster2, cluster2MeanThreshold)
coreClusterwiseMeanCluster3, boundaryClusterwiseMeanCluster3 = findingCore_boundary_MeanThreshold(cluster3, cluster3MeanThreshold)


#Seperating data points into core and boundary categoris based on clusterWiseMedianThreshold value
coreClusterwiseMedianCluster1, boundaryClusterwiseMedianCluster1 = findingCore_boundary_MedianThreshold(cluster1, cluster1MedianThreshold)
coreClusterwiseMedianCluster2, boundaryClusterwiseMedianCluster2 = findingCore_boundary_MedianThreshold(cluster2, cluster2MedianThreshold)
coreClusterwiseMedianCluster3, boundaryClusterwiseMedianCluster3 = findingCore_boundary_MedianThreshold(cluster3, cluster3MedianThreshold)

def mean_classification_using_KNN(coreMeanCluster1, coreMeanCluster2, coreMeanCluster3, boundaryMeanCluster1, boundaryMeanCluster2, boundaryMeanCluster3):
    # Assign labels to each core cluster
    coreMeanCluster1_c5 = np.full((len(coreMeanCluster1), 1), 1.0)
    coreMeanCluster2_c5 = np.full((len(coreMeanCluster2), 1), 2.0)
    coreMeanCluster3_c5 = np.full((len(coreMeanCluster3), 1), 3.0)

    # Combine the core cluster data with their labels
    coreMeanCluster1_m = np.hstack((coreMeanCluster1, coreMeanCluster1_c5))
    coreMeanCluster2_m = np.hstack((coreMeanCluster2, coreMeanCluster2_c5))
    coreMeanCluster3_m = np.hstack((coreMeanCluster3, coreMeanCluster3_c5))

    # Concatenate all core clusters into one dataset
    coreMeanCluster = np.vstack((coreMeanCluster1_m, coreMeanCluster2_m, coreMeanCluster3_m))

    # Prepare data for KNN
    X_core = coreMeanCluster[:, :-1]  # Features
    y_core = coreMeanCluster[:, -1]   # Labels

    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the KNN classifier with core data
    knn.fit(X_core, y_core)

    # Initialize lists to store combined boundary data and their predictions
    X_boundary_combined = []
    y_boundary_combined = []

    # Function to process each boundary cluster
    def process_boundary_cluster(boundary_cluster, label):
        if len(boundary_cluster) > 0:
            # Predict the labels for boundary data
            y_boundary_pred = knn.predict(boundary_cluster)
            X_boundary_combined.append(boundary_cluster)
            y_boundary_combined.extend(y_boundary_pred)

    # Process each boundary cluster
    process_boundary_cluster(boundaryMeanCluster1, 1.0)
    process_boundary_cluster(boundaryMeanCluster2, 2.0)
    process_boundary_cluster(boundaryMeanCluster3, 3.0)

    # Combine core and boundary data for final dataset
    if len(X_boundary_combined) > 0:
        X_boundary_combined = np.vstack(X_boundary_combined)
    else:
        X_boundary_combined = np.empty((0, X_core.shape[1]))

    core_data_df = pd.DataFrame(coreMeanCluster)
    boundary_data_df = pd.DataFrame(np.hstack((X_boundary_combined, np.array(y_boundary_combined).reshape(-1, 1))))
    concat_df = pd.concat([core_data_df, boundary_data_df], ignore_index=True)

    # Remove label column before computing XB index
    new_df = concat_df.iloc[:, :-1].to_numpy()
    # Combine core and boundary cluster labels
    core_cluster_df = pd.DataFrame(y_core)
    boundary_cluster_df = pd.DataFrame(y_boundary_combined)
    cluster = pd.concat([core_cluster_df, boundary_cluster_df], ignore_index=True)

    # Convert cluster labels to NumPy array for compatibility
    cluster = cluster.to_numpy().flatten()
    return (new_df, cluster)


def median_classification_using_KNN(coreMedianCluster1, coreMedianCluster2, coreMedianCluster3, boundaryMedianCluster1, boundaryMedianCluster2, boundaryMedianCluster3):
    # Assign labels to each core cluster
    coreMedianCluster1_c5 = np.full((len(coreMedianCluster1), 1), 1.0)
    coreMedianCluster2_c5 = np.full((len(coreMedianCluster2), 1), 2.0)
    coreMedianCluster3_c5 = np.full((len(coreMedianCluster3), 1), 3.0)

    # Combine the core cluster data with their labels
    coreMedianCluster1_m = np.hstack((coreMedianCluster1, coreMedianCluster1_c5))
    coreMedianCluster2_m = np.hstack((coreMedianCluster2, coreMedianCluster2_c5))
    coreMedianCluster3_m = np.hstack((coreMedianCluster3, coreMedianCluster3_c5))

    # Concatenate all core clusters into one dataset
    coreMedianCluster = np.vstack((coreMedianCluster1_m, coreMedianCluster2_m, coreMedianCluster3_m))

    # Prepare data for KNN
    X_core = coreMedianCluster[:, :-1]  # Features
    y_core = coreMedianCluster[:, -1]   # Labels

    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the KNN classifier with core data
    knn.fit(X_core, y_core)

    # Initialize lists to store combined boundary data and their predictions
    X_boundary_combined = []
    y_boundary_combined = []

    # Function to process each boundary cluster
    def process_boundary_cluster(boundary_cluster, label):
        if len(boundary_cluster) > 0:
            # Predict the labels for boundary data
            y_boundary_pred = knn.predict(boundary_cluster)
            X_boundary_combined.append(boundary_cluster)
            y_boundary_combined.extend(y_boundary_pred)

    # Process each boundary cluster
    process_boundary_cluster(boundaryMedianCluster1, 1.0)
    process_boundary_cluster(boundaryMedianCluster2, 2.0)
    process_boundary_cluster(boundaryMedianCluster3, 3.0)

    # Combine core and boundary data for final dataset
    if len(X_boundary_combined) > 0:
        X_boundary_combined = np.vstack(X_boundary_combined)
    else:
        X_boundary_combined = np.empty((0, X_core.shape[1]))

    core_data_df = pd.DataFrame(coreMedianCluster)
    boundary_data_df = pd.DataFrame(np.hstack((X_boundary_combined, np.array(y_boundary_combined).reshape(-1, 1))))
    concat_df = pd.concat([core_data_df, boundary_data_df], ignore_index=True)

    # Remove label column before computing XB index
    new_df = concat_df.iloc[:, :-1].to_numpy()
    # Combine core and boundary cluster labels
    core_cluster_df = pd.DataFrame(y_core)
    boundary_cluster_df = pd.DataFrame(y_boundary_combined)
    cluster = pd.concat([core_cluster_df, boundary_cluster_df], ignore_index=True)

    # Convert cluster labels to NumPy array for compatibility
    cluster = cluster.to_numpy().flatten()
    return (new_df, cluster)


def compute_xb_index(data, labels, centroids):
    k = len(centroids)
    n = len(data)

    # Calculate the distance matrix
    distances = pairwise_distances_argmin_min(data, centroids)[1]

    # Calculate the sum of the within-cluster distances
    within_cluster_distances = np.sum(distances)

    cluster_dispersion = 0
    for i in range(k):
        cluster_points = data[np.where(labels == i)[0]]

        # Check for empty cluster and skip if necessary
        if cluster_points.shape[0] == 0:
            continue

        cluster_dispersion += np.sum(pairwise_distances_argmin_min(cluster_points, [centroids[i]])[1])

    # Calculate the Xie-Beni index
    xie_beni = within_cluster_distances / (k * cluster_dispersion / n)

    return xie_beni


#########################################################################
#For single mean
new_df,cluster=mean_classification_using_KNN(coreSingleMeanCluster1,coreSingleMeanCluster2,coreSingleMeanCluster3,boundarySingleMeanCluster1,boundarySingleMeanCluster2,boundarySingleMeanCluster1)
# Compute the XB index
xb_index = compute_xb_index(new_df, cluster, Centers)
print("Xie-Beni Index for single Mean:", xb_index)
#########################################################################
#For cluster wise mean
new_df,cluster=mean_classification_using_KNN(coreClusterwiseMeanCluster1,coreClusterwiseMeanCluster2,coreClusterwiseMeanCluster3,boundaryClusterwiseMeanCluster1,boundaryClusterwiseMeanCluster2,boundaryClusterwiseMeanCluster3)
# Compute the XB index
xb_index = compute_xb_index(new_df, cluster, Centers)
print("Xie-Beni Index for Cluster wise Mean:", xb_index)

#########################################################################
##For single median
new_df,cluster=median_classification_using_KNN(coreSingleMedianCluster1,coreSingleMedianCluster2,coreSingleMedianCluster3,boundarySingleMedianCluster1,boundarySingleMedianCluster2,boundarySingleMedianCluster3)
# Compute the XB index
xb_index = compute_xb_index(new_df, cluster, Centers)
print("Xie-Beni Index for single Median:", xb_index)

#########################################################################
#For cluster wise median

new_df,cluster=median_classification_using_KNN(coreClusterwiseMedianCluster1,coreClusterwiseMedianCluster2,coreClusterwiseMedianCluster3,boundaryClusterwiseMedianCluster1,boundaryClusterwiseMedianCluster2,boundaryClusterwiseMedianCluster3)
# Compute the XB index
xb_index = compute_xb_index(new_df, cluster, Centers)
print("Xie-Beni Index for cluster wise Median:", xb_index)

