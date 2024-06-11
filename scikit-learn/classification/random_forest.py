from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
clf.fit(X_train, y_train)

# Make predictions on your test data
y_pred = clf.predict(X_test)
