import numpy as np
from sklearn.linear_model import LinearRegression

# Sample input data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict using the trained model
X_test = np.array([[6], [7], [8]])
y_pred = model.predict(X_test)

# Print the predicted values
print(y_pred)
