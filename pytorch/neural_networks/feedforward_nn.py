import torch

import torch.nn as nn


# Define the neural network architecture
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Define the input size, hidden size, and output size
input_size = 10
hidden_size = 20
output_size = 5

# Create an instance of the feedforward neural network
model = FeedForwardNN(input_size, hidden_size, output_size)

# Create some random input data
input_data = torch.randn(32, input_size)

# Pass the input data through the model
output = model(input_data)

# Print the output
print(output)
