from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

#@ DataPoints
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

X=np.array(list(zip(x1, x2))).reshape(len(x1), 2)

# K means determine k
distortions=[]

K= range(1,10)

for k in K:
    KMeansModel=KMeans(n_clusters=k).fit(X)
    KMeansModel.fit(X)
    distortions.append(sum(np.min(cdist(X, KMeansModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])



#@ Visualization

plt.plot(K, distortions, 'bx-')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
