import numpy as np

# Set the seed for reproducibility (optional)
np.random.seed(0)

# Generate an array of 10 random integers between 0 and 9
random_integers = np.random.randint(0, 10, size=10)

# Generate an array of 10 random floats between 0 and 1
random_floats = np.random.random(size=10)

# Generate a 2x3 array of random numbers from a standard normal distribution
random_normal = np.random.randn(2, 3)

# Print the generated arrays
print("Random Integers:", random_integers)
print("Random Floats:", random_floats)
print("Random Normal:", random_normal)
