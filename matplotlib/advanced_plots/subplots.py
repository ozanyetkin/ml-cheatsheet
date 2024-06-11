import matplotlib.pyplot as plt

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 2)

# Plot data on the subplots
axs[0, 0].plot([1, 2, 3, 4], [1, 4, 2, 3])
axs[0, 0].set_title("Subplot 1")

axs[0, 1].plot([1, 2, 3, 4], [1, 4, 2, 3])
axs[0, 1].set_title("Subplot 2")

axs[1, 0].plot([1, 2, 3, 4], [1, 4, 2, 3])
axs[1, 0].set_title("Subplot 3")

axs[1, 1].plot([1, 2, 3, 4], [1, 4, 2, 3])
axs[1, 1].set_title("Subplot 4")

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
