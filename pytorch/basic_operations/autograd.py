import torch

# Create tensors with requires_grad=True to track computation
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

# Perform some operations
z = x + y
w = z * 2

# Compute gradients
w.backward()

# Print gradients
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)
