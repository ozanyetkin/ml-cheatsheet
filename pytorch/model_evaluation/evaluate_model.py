import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch.nn as nn
import torch.optim as optim


# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


# Create some example data
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Instantiate your model
model = MyModel()

# Define your loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), y.float())
    loss.backward()
    optimizer.step()

# Evaluation
with torch.no_grad():
    model.eval()
    test_outputs = model(X)
    test_predictions = torch.round(torch.sigmoid(test_outputs)).squeeze().numpy()

# Calculate evaluation metrics
accuracy = accuracy_score(y.numpy(), test_predictions)
precision = precision_score(y.numpy(), test_predictions)
recall = recall_score(y.numpy(), test_predictions)
f1 = f1_score(y.numpy(), test_predictions)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
