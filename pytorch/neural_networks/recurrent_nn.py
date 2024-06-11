import torch

import torch.nn as nn


# Define the recurrent neural network model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = torch.zeros(1, input.size(0), self.hidden_size)
        output, _ = self.rnn(input, hidden)
        output = self.fc(output[-1])
        return output


# Define the input and output sizes
input_size = 10
hidden_size = 20
output_size = 5

# Create an instance of the RNN model
rnn = RNN(input_size, hidden_size, output_size)

# Create a random input sequence
input_sequence = torch.randn(3, 1, input_size)

# Pass the input sequence through the RNN
output = rnn(input_sequence)

# Print the output
print(output)
