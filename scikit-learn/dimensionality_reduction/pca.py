import numpy as np
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import PCA

# Load your dataset
X = ...

# Create an instance of PCA
pca = PCA(n_components=2)

# Fit the data to the PCA model
pca.fit(X)

# Transform the data to the lower-dimensional space
X_transformed = pca.transform(X)

# Access the principal components
principal_components = pca.components_

# Access the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Access the singular values
singular_values = pca.singular_values_

# Print the results
print("Transformed data:")
print(X_transformed)
print("Principal components:")
print(principal_components)
print("Explained variance ratio:")
print(explained_variance_ratio)
print("Singular values:")
print(singular_values)
