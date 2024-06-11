import matplotlib.pyplot as plt

# Data for the x-axis
x = ["A", "B", "C", "D", "E"]

# Data for the y-axis
y = [10, 15, 7, 12, 9]

# Create a bar chart
plt.bar(x, y)

# Add labels and title
plt.xlabel("Categories")
plt.ylabel("Values")
plt.title("Bar Chart Example")

# Display the chart
plt.show()
