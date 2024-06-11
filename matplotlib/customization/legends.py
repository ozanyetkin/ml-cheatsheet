import matplotlib.pyplot as plt

# Create some data
x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [1, 8, 27, 64, 125]

# Plot the data
plt.plot(x, y1, label="Line 1")
plt.plot(x, y2, label="Line 2")

# Add a legend
plt.legend()

# Show the plot
plt.show()
