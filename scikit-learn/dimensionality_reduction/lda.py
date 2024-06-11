from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def lda_dimensionality_reduction(X, y, n_components):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    return X_lda


# Example usage
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 0, 1, 1]
n_components = 1

X_lda = lda_dimensionality_reduction(X, y, n_components)
print(X_lda)
