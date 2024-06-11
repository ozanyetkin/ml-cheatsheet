import numpy as np

import matplotlib.pyplot as plt

# Generate some data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y1, label="sin(x)")
ax.plot(x, y2, label="cos(x)")

# Set the title and labels
ax.set_title("Trigonometric Functions")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Customize the plot
ax.grid(True)  # Add grid lines
ax.axhline(0, color="black", linewidth=0.5)  # Add a horizontal line at y=0
ax.axvline(5, color="red", linestyle="--", linewidth=0.5)  # Add a vertical line at x=5

# Add text annotation
ax.text(2, 0.5, "Maximum", fontsize=10, ha="center")
ax.annotate("Minimum", xy=(7, -0.5), xytext=(8, -0.8), arrowprops=dict(arrowstyle="->"))

# Add a legend
ax.legend()

# Save the plot as an image
plt.savefig("overview.png")

# Show the plot
plt.show()
