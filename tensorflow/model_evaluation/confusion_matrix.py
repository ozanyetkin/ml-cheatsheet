import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

# Generate some example predictions and ground truth labels
predictions = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
labels = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])

# Create a confusion matrix using TensorFlow
confusion = tf.math.confusion_matrix(labels, predictions)

# Convert the TensorFlow tensor to a NumPy array
confusion = confusion.numpy()

# Plot the confusion matrix
plt.imshow(confusion, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.colorbar()
plt.show()
