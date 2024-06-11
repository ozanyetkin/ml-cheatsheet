import torch

import torch.nn as nn
import torch.optim as optim


# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# Create an instance of your model
model = MyModel()

# Define your loss function
criterion = nn.MSELoss()

# Define your training data
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# Define your optimizers
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01)
adam_optimizer = optim.Adam(model.parameters(), lr=0.01)
rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Zero the gradients
    sgd_optimizer.zero_grad()
    adam_optimizer.zero_grad()
    rmsprop_optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    loss.backward()

    # Update the parameters using different optimizers
    sgd_optimizer.step()
    adam_optimizer.step()
    rmsprop_optimizer.step()

    # Print the loss for each optimizer
    print(
        f"Epoch {epoch+1}: SGD Loss = {loss.item()}, Adam Loss = {loss.item()}, RMSprop Loss = {loss.item()}"
    )
