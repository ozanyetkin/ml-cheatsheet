from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Define the parameter grid
param_grid = {
    "C": [0.1, 1, 10],
    "gamma": [0.1, 0.01, 0.001],
    "kernel": ["linear", "rbf"],
}

# Create the SVM classifier
svm = SVC()

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)

# Fit the data to find the best parameters
grid_search.fit(iris.data, iris.target)

# Print the best parameters and the best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
