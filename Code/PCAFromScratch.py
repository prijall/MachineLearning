import numpy as np

class PCA:
  def __init__(self, n_components):
    self.n_components=n_components

  def fit(self, X):
    # Standardization of data:
    X=X.copy()
    self.mean=np.mean(X, axis=0)
    self.scale=np.std(X, axis=0)
    X_std=(X-self.mean)/ self.scale

     # Eigen Decomposition of Covariance matrix:
    cov_mat=np.cov(X_std.T)
    eigen_values, eigen_vectors=np.linalg.eig(cov_mat)

    # Adjusting the eigen vectors that largest in absolute value to be positive:
    max_abs_idx=np.argmax(np.abs(eigen_vectors), axis=0)
    signs=np.sign(eigen_vectors[max_abs_idx, range(eigen_vectors.shape[0])])
    eigen_vectors=eigen_vectors*signs[np.newaxis, :]
    eigen_vectors=eigen_vectors.T

    eigen_pairs=[(np.abs(eigen_values[i]), eigen_vectors[i, :]) for i in range(len(eigen_values))]
    eigen_pairs.sort(key=lambda x:x[0], reverse=True)
    eigen_values_sorted=np.array([X[0] for X in eigen_pairs])
    eigen_vectors_sorted=np.array([X[1] for X in eigen_pairs])

    self.components=eigen_vectors_sorted[:self.n_components, :]


    # Explained variance ratio:
    self.explained_variance_ratio=[i/np.sum(eigen_values) for i in eigen_values_sorted[:self.n_components]]
    self.cum_explained_variance=np.cumsum(self.explained_variance_ratio)

    return self
  
  def transform(self, X):
        X = X.copy()
        X_std = (X - self.mean) / self.scale
        X_proj = X_std.dot(self.components.T)
        
        return X_proj

   
#@ Testing:

from sklearn.datasets import load_iris

iris=load_iris()
X=iris['data']
y=iris['target']

n_samples, n_features=X.shape
print('Number of samples:', n_samples)
print('Number of features:', n_features)

my_pca=PCA(n_components=2).fit(X)
X_proj=my_pca.transform(X)
print('Transformed data shape from scratch:', X_proj.shape)