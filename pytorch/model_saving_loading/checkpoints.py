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

# Define your optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Train your model
for epoch in range(10):
    # Perform forward pass
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    outputs = model(inputs)

    # Calculate loss
    loss = criterion(outputs, targets)

    # Perform backward pass and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save model checkpoint
    if (epoch + 1) % 5 == 0:
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch + 1}.pt")
