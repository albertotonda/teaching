# Example of dimensionality reduction and clustering, using Iris
# by Alberto Tonda, 2019 <alberto.tonda@gmail.com>

# creating this vector will be useful for later
colors = ['blue', 'orange', 'red', 'black', 'green', 'yellow']

# load Iris dataset
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

import numpy as np
print("Iris dataset has %d samples and %d features, with %d classes" % (X.shape[0], X.shape[1], len(np.unique(y))))

# let's try to reduce the dataset to two dimensions using PCA
print("Computing Principal Component Analysis...")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# plot the result
import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1]) 
plt.title("PCA results")
plt.show()

# again, but this time let's separate classes by color
for i in range(0, len(np.unique(y))) :
    plt.scatter(X[y==i,0], X[y==i,1], color=colors[i])
plt.title("PCA results, colored by class")
plt.show()

# now, let's try some clustering!
from sklearn.cluster import DBSCAN
clusterer = DBSCAN(eps=0.5)
clusterer.fit(X)

# and plot the results
cluster_labels = clusterer.labels_

for cluster in np.unique(cluster_labels) :
    plt.scatter(X[cluster_labels==cluster,0], X[cluster_labels==cluster,1], color=colors[cluster])
plt.title("Clustering results")
plt.show()
