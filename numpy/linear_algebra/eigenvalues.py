import numpy as np

# Define a matrix
matrix = np.array([[1, 2], [3, 4]])

# Calculate the eigenvalues
eigenvalues = np.linalg.eigvals(matrix)

# Display the eigenvalues
print(
    "Eigenvalues are the values λ for which the equation Av = λv holds, where A is the matrix, v is the eigenvector, and λ is the eigenvalue."
)
print("Eigenvalues:", eigenvalues)
