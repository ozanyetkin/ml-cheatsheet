import matplotlib.pyplot as plt

# Create some data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a plot
plt.plot(x, y)

# Add an annotation
plt.annotate(
    "Max Value",
    xy=(5, 10),
    xytext=(4, 8),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
)

# Add labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Annotations Example")

# Show the plot
plt.show()
