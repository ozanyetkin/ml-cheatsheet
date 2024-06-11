import torch

# Create two tensors
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# Addition
addition = tensor1 + tensor2
print("Addition:")
print(addition)

# Subtraction
subtraction = tensor1 - tensor2
print("Subtraction:")
print(subtraction)

# Multiplication
multiplication = tensor1 * tensor2
print("Multiplication:")
print(multiplication)

# Division
division = tensor1 / tensor2
print("Division:")
print(division)

# Matrix multiplication
matrix_multiplication = torch.matmul(tensor1, tensor2)
print("Matrix Multiplication:")
print(matrix_multiplication)

# Element-wise square root
sqrt = torch.sqrt(tensor1)
print("Square Root:")
print(sqrt)
