#Required libraries


import itertools
from scipy.spatial import distance
import numpy as np
import math

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, Birch
import warnings
from sklearn.metrics import normalized_mutual_info_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering

from metafeatures import Meta
from deap import base
from deap import creator, tools
from scipy.stats import spearmanr
from cvi import Validation
from sklearn.preprocessing import OrdinalEncoder
import glob
from scipy.io import arff
import pandas as pd
from cvi import Validation
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from itertools import combinations

from sklearn.neighbors import NearestNeighbors

"""
This python file includes the definition of a few of the Cluster Validity Indices (CVI) methods used in the experiment. The CVI found here are
c_index, i_index, banfeld_raferty. These methods are original from cSMartML package. 

"""

def c_index(data, class_label):
    """
        The C-Index, a measure of dispersionn
    """
    sw = 0
    nw = 0
    numCluster = max(class_label) + 1
    data_matrix = np.asmatrix(data).astype(np.float)
    
    # iterate through all the clusters
    for i in range(numCluster):
        indices = [t for t, x in enumerate(class_label) if x == i]
        clusterMember = data_matrix[indices, :]
        # compute distance of every pair of points
        list_clusterDis = distance.pdist(clusterMember)
        sw = sw + sum(list_clusterDis)
        nw = nw + len(list_clusterDis)
    # compute the pairwise distance of the whole dataset
    list_dataDis = distance.pdist(data_matrix)
    # compute smin
    sortedList = sorted(list_dataDis)
    smin = sum(sortedList[0:nw])
    # compute smax
    sortedList = sorted(list_dataDis, reverse=True)
    smax = sum(sortedList[0:nw])
    
    # compute the score
    return (sw - smin) / (smax - smin)


#i-index CVI  
def i_index(data, class_label):
      """
        The I index, a measure of compactness.
      """
      
      normClusterSum = 0
      normDatasetSum = 0
      list_centers = []

      data_matrix = np.asmatrix(data).astype(np.float)
      # compute the number of clusters and attribute
      attributes = len(data_matrix[0])
      numCluster = max(class_label) + 1
      # compute the center of the dataset
      dataCenter = np.mean(data_matrix, 0)
      # iterate through all the clusters
      for i in range(numCluster):
          indices = [t for t, x in enumerate(class_label) if x == i]
          clusterMember = data_matrix[indices, :]
          # compute the center of the cluster
          clusterCenter = np.mean(clusterMember, 0)
          list_centers.append(np.asarray(clusterCenter))
          # compute the norm for every member in the cluster with cluster center and dataset center
          for member in clusterMember:
              normClusterSum += distance.euclidean(member, clusterCenter)
              normDatasetSum += distance.euclidean(member, dataCenter)
      # compute the max distance between cluster centers
      list_centers = np.concatenate(list_centers, axis=0)
      maxCenterDis = max(distance.pdist(list_centers))
      
      # compute the fitness      
      return math.pow(((normDatasetSum * maxCenterDis) / (normClusterSum * numCluster)), attributes)
    

#Banfel Rafery CVI
def banfeld_raferty(data, class_label):
    """ Banfeld-Raferty index is the weighted sum of the logarithms
         of the traces of the variance-covariance matrix of each cluster
         
        Weighted sum of the logarithms of the traces of the variance-covariance matrix of each cluster
        
        OBJECTIVE: MIN
    """

    sum_total = 0
    num_cluster = max(class_label) + 1
    data_matrix = np.asmatrix(data).astype(np.float)
    
    # iterate through all the clusters
    for i in range(num_cluster):
        sum_dis = 0
        indices = [t for t, x in enumerate(class_label) if x == i]
        cluster_member = data_matrix[indices, :]

        # compute the center of the cluster
        cluster_center = np.mean(cluster_member, 0)

        # iterate through all the members
        for member in cluster_member:
            sum_dis += distance.euclidean(member, cluster_center) ** 2

        op = sum_dis / len(indices)
        if op <= 0:
            # Cannot calculate Banfeld_Raferty, due to an undefined value
            continue
        else:
            sum_total += len(indices) * math.log(sum_dis / len(indices))

    return sum_total

