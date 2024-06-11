from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a logistic regression model
model = LogisticRegression()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Print the accuracy scores for each fold
for fold, score in enumerate(scores):
    print(f"Fold {fold+1}: {score}")

# Print the mean accuracy score
print("Mean Accuracy:", scores.mean())
