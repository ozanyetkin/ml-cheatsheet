import numpy as np

import matplotlib.pyplot as plt

# Generate some random data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Set the plot style
plt.style.use("ggplot")

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the data with different styles
ax.plot(x, y1, label="sin(x)", linestyle="-", linewidth=2)
ax.plot(x, y2, label="cos(x)", linestyle="--", linewidth=2)

# Add labels and a legend
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Plot Styles Example")
ax.legend()

# Show the plot
plt.show()
