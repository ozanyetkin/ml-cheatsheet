import numpy as np

import matplotlib.pyplot as plt

# Create a random 2D array
data = np.random.rand(10, 10)

# Create a figure and axis
fig, ax = plt.subplots()

# Create the heatmap
heatmap = ax.imshow(data, cmap="hot")

# Add colorbar
cbar = plt.colorbar(heatmap)

# Set labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Heatmap Example")

# Show the plot
plt.show()
