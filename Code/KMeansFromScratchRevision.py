import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

def euclidean(point, data):
    return np.sqrt(np.sum((point-data)**2, axis=1))


class KMeans:
    def __init__(self, n_cluster=5, max_iter=200):
        self.n_cluster=n_cluster
        self.max_iter=max_iter

    def fit(self, X_train):
        #@ Initializing the centriods, where a random datapoint is selected as the first
        self.centriods=[random.choice(X_train)]

        for _ in range(self.n_cluster-1):
            #@ Calculate distance from Centriods
            dists=np.sum([euclidean(centriod, X_train) for centriod in self.centriods], axis=0)
            
            #Normalize the distances:
            dists /=np.sum(dists)

            #Choosing remaining points based on their distances
            new_centriod_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)


        iteration=0
        prev_centriods=None
        while np.not_equal(self.centriods, prev_centriods).any() and iteration < self.max_iter:
            #Sort each datapoint, assigning to nearest centriods
            sorted_points=[[] for _ in range(self.n_cluster)]
            for x in X_train:
                dists=euclidean(x, self.centriods)
                centriods_idx=np.argmin(dists)
                sorted_points[centriods_idx].append(x)
    
    #Push current centriods to previous, reassign centriods as mean of points belonging to them
            prev_centriods=self.centriods
            self.centriods=[np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centriod in enumerate(self.centriods):
                if np.isnan(centriod).any():
                    self.centriods[i]=prev_centriods[i]
            iteration += 1


    def evaluate(self, X):
        centriods=[]
        centriods_idxs=[]

        for x in X:
            dists=euclidean(x, self.Centriods)
            centriods_idx=np.argmin(dists)
            centriods.append(self.centriods[centriods_idx])
            centriods_idxs.append(centriods_idx)
            
        return centriods, centriods_idxs
     


