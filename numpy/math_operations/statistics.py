import numpy as np

# Create a random array of numbers
data = np.random.randint(0, 100, size=10)

# Calculate the mean
mean = np.mean(data)
print("Mean:", mean)

# Calculate the median
median = np.median(data)
print("Median:", median)

# Calculate the standard deviation
std_dev = np.std(data)
print("Standard Deviation:", std_dev)

# Calculate the variance
variance = np.var(data)
print("Variance:", variance)

# Calculate the minimum and maximum values
min_value = np.min(data)
max_value = np.max(data)
print("Minimum Value:", min_value)
print("Maximum Value:", max_value)
