import numpy as np

import matplotlib.pyplot as plt

# Generate some random data
data = np.random.randn(1000)

# Create a histogram
plt.hist(data, bins=30, edgecolor="black")

# Set labels and title
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram")

# Show the plot
plt.show()
