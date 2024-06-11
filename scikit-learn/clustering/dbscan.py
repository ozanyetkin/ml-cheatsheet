from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=200, centers=3, random_state=0)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering")
plt.show()
