import torch

import torch.nn as nn
import torch.optim as optim


# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# Create an instance of the model
model = MyModel()

# Define some dummy data
input_data = torch.randn(1, 10)

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(10):
    output = model(input_data)
    loss = criterion(output, torch.tensor([1.0]))  # Dummy target value
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), "saved_model.pth")

# Load the model
loaded_model = MyModel()
loaded_model.load_state_dict(torch.load("saved_model.pth"))
loaded_model.eval()

# Use the loaded model for inference
output = loaded_model(input_data)
print(output)
