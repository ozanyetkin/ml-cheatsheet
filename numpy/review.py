import numpy as np

# Creating an array
arr = np.array([1, 2, 3, 4, 5])

# Accessing elements
print("First element:", arr[0])
print("Last element:", arr[-1])
print("Slice of array:", arr[2:4])

# Array shape and size
print("Shape of array:", arr.shape)
print("Size of array:", arr.size)

# Reshaping an array
reshaped_arr = arr.reshape(5, 1)
print("Reshaped array:\n", reshaped_arr)

# Arithmetic operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print("Sum of arrays:", np.add(arr1, arr2))
print("Product of arrays:", np.multiply(arr1, arr2))

# Statistical operations
print("Mean of array:", np.mean(arr))
print("Standard deviation of array:", np.std(arr))
print("Maximum value of array:", np.max(arr))
print("Minimum value of array:", np.min(arr))

# Generating random numbers
random_arr = np.random.rand(3, 3)
print("Random array:\n", random_arr)

# Array concatenation
concatenated_arr = np.concatenate((arr1, arr2))
print("Concatenated array:", concatenated_arr)

# Sorting an array
sorted_arr = np.sort(arr)
print("Sorted array:", sorted_arr)
