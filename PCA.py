import numpy as np

class FromScratchPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        X_c = X - self.mean_
        N, D = X_c.shape
        
        max_components = min(N, D)
        if self.n_components is None:
            self.n_components_ = max_components
        else:
            self.n_components_ = self.n_components

        L = (X_c @ X_c.T) / N
        
        e_values, e_vectors_v = np.linalg.eig(L)

        # Filter positive eigenvalues for stability
        positive_mask = e_values > 1e-10
        e_values = e_values[positive_mask]
        e_vectors_v = e_vectors_v[:, positive_mask]

        U_unnormalized = X_c.T @ e_vectors_v
        
        U = U_unnormalized / np.sqrt(N*e_values)
        
        self.components_ = U.T[:self.n_components_]
        self.explained_variance_ = e_values[:self.n_components_]
        
        return self

    def transform(self, X: np.ndarray):
        if self.mean_ is None:
            raise RuntimeError("The estimator must be fitted before transforming data.")
        
        X_c = X - self.mean_
        X_transformed = X_c @ self.components_.T
        
        return X_transformed

    def fit_transform(self, X: np.ndarray, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray):
        X_reconstructed = (X_transformed @ self.components_) + self.mean_
        return X_reconstructed