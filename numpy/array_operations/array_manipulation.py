import numpy as np

# Create a 1-dimensional array
arr = np.array([1, 2, 3, 4, 5])

# Reshape the array into a 2x3 matrix
reshaped_arr = arr.reshape(2, 3)

# Transpose the matrix
transposed_arr = reshaped_arr.T

# Flatten the matrix into a 1-dimensional array
flattened_arr = transposed_arr.flatten()

# Concatenate two arrays vertically
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
vertical_concatenated_arr = np.vstack((arr1, arr2))

# Concatenate two arrays horizontally
horizontal_concatenated_arr = np.hstack((arr1, arr2))

# Split the array into multiple sub-arrays
split_arr = np.array([1, 2, 3, 4, 5, 6])
sub_arrays = np.split(split_arr, 3)

# Print the results
print("Original array:")
print(arr)
print("\nReshaped array:")
print(reshaped_arr)
print("\nTransposed array:")
print(transposed_arr)
print("\nFlattened array:")
print(flattened_arr)
print("\nVertically concatenated array:")
print(vertical_concatenated_arr)
print("\nHorizontally concatenated array:")
print(horizontal_concatenated_arr)
print("\nSplit arrays:")
for sub_arr in sub_arrays:
    print(sub_arr)
