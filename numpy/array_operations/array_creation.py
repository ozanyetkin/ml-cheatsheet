import numpy as np

# Create a 1-dimensional array
arr1 = np.array([1, 2, 3, 4, 5])
print("1D Array:")
print(arr1)

# Create a 2-dimensional array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:")
print(arr2)

# Create an array of zeros
zeros_arr = np.zeros((3, 4))
print("Array of Zeros:")
print(zeros_arr)

# Create an array of ones
ones_arr = np.ones((2, 3))
print("Array of Ones:")
print(ones_arr)

# Create an array with a range of values
range_arr = np.arange(1, 10, 2)
print("Array with Range of Values:")
print(range_arr)

# Create a random array
random_arr = np.random.rand(3, 2)
print("Random Array:")
print(random_arr)
