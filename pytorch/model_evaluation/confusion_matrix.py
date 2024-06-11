import torch
from sklearn.metrics import confusion_matrix

# Assuming you have predicted labels and ground truth labels
predicted_labels = torch.tensor([0, 1, 1, 0, 2, 1])
ground_truth_labels = torch.tensor([0, 1, 2, 0, 2, 1])

# Convert tensors to numpy arrays
predicted_labels = predicted_labels.numpy()
ground_truth_labels = ground_truth_labels.numpy()

# Calculate confusion matrix
cm = confusion_matrix(ground_truth_labels, predicted_labels)

print(cm)
