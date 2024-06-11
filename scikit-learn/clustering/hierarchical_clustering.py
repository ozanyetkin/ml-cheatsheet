import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

# Generate some random data
X, y = make_blobs(n_samples=100, centers=3, random_state=0)

# Create an instance of the AgglomerativeClustering class
clustering = AgglomerativeClustering(n_clusters=3)

# Fit the model to the data
clustering.fit(X)

# Get the predicted labels
labels = clustering.labels_

# Plot the data points with different colors for each cluster
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.show()
