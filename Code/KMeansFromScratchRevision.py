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
        self.Centriods=[random.choice(X_train)]

        for _ in range(self.n_cluster-1):
            #@ Calculate distance from Centriods

            dists=np.sum([euclidean(Centriod, X_train) for Centriod in self.Centriods], axis=0)
            

            #Normalize the distances:
            dists /=np.sum(dists)

            #Choosing remaining points based on their distances
            new_centriods_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)


        iteration=0
        prev_centriods=None

