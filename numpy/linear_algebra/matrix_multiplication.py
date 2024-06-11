import numpy as np

# Define two matrices
matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
matrix2 = np.array([[7, 8], [9, 10], [11, 12]])

# Perform matrix multiplication using np.dot()
result = np.dot(matrix1, matrix2)

# Print the matrices and the result
print("Matrix 1:")
print(matrix1)
print("Matrix 2:")
print(matrix2)
print("Result of matrix multiplication:")
print(result)
