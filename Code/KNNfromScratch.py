import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#@ Sample dataset:
data=np.array([
    [2, 4, 0],
    [4, 2, 0], 
    [4, 4, 0],
    [6, 4, 0], 
    [6, 6, 1],
    [8, 6, 1]
])

#@ Spliting features and labels:
X, y=data[:, :2], data[:, -1]

#@ Implementing Euclidean distance:
def Euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1-point2)**2))