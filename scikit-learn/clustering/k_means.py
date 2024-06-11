import numpy as np
from sklearn.cluster import KMeans

# Generate some random data
X = np.random.rand(100, 2)

# Create a KMeans object with the desired number of clusters
kmeans = KMeans(n_clusters=3)

# Fit the data to the KMeans model
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the coordinates of the cluster centers
centers = kmeans.cluster_centers_

# Print the cluster labels and centers
print("Cluster Labels:")
print(labels)
print("Cluster Centers:")
print(centers)
