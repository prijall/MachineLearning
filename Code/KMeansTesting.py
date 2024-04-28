#@ Model Implementation
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
from KMeansFromScratchRevision import KMeans 
import seaborn as sns
import matplotlib.pyplot as plt

#@ Creating the dataset of 2D distribution

k=4  #no of centriods

X_train, true_labels=make_blobs(n_samples=100, centers=k, random_state=42)
X_train=StandardScaler().fit_transform(X_train)

#@ Fitting centriods(k) to dataset
Kmn=KMeans(n_cluster=k)
Kmn.fit(X_train)

#@ Viewing Result

class_center, classification=Kmn.evaluate(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=true_labels,
                style=classification,
                palette='deep',
                legend=None)

plt.plot([x for x, _ in Kmn.centriods],
         [y for _, y in Kmn.centriods],
         markersize=10)
plt.show()