import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(42)
n_points = 100
x = np.random.rand(n_points)
y = np.random.rand(n_points)
z = np.random.rand(n_points)

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the data points
ax.scatter(x, y, z, c="b", marker="o")

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Scatter Plot")

# Show the plot
plt.show()
