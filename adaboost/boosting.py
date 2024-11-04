# code was used from https://github.com/srijarkoroy/adaboost
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from datagenerate import dataset
from plot import plot_adaboost

class AdaBoost:
    def __init__(self):
        self.stumps = []
        self.stump_weights = []
        self.errors = []
        self.sample_weights = []
        self.ada_errors = []
        self.training_errors = []  # To store training error at each iteration

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int):
        n = X.shape[0]
        # Initialize weights uniformly
        sample_weights = np.ones(n) / n
        self.sample_weights.append(sample_weights.copy())

        for t in range(iters):
            # Fit weak learner
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump = stump.fit(X, y, sample_weight=sample_weights)

            # Predict and calculate error
            stump_pred = stump.predict(X)
            err = np.sum(sample_weights * (stump_pred != y)) / np.sum(sample_weights)
            # Clip err to prevent division by zero
            err = np.clip(err, 1e-10, 1 - 1e-10)
            self.errors.append(err)

            # Calculate stump weight
            stump_weight = 0.5 * np.log((1 - err) / err)
            self.stump_weights.append(stump_weight)

            # Update sample weights
            # Convert labels to {-1, 1}
            y_binary = np.where(y == 1, 1, -1)
            stump_pred_binary = np.where(stump_pred == 1, 1, -1)
            sample_weights = sample_weights * np.exp(-stump_weight * y_binary * stump_pred_binary)
            sample_weights /= np.sum(sample_weights)
            self.sample_weights.append(sample_weights.copy())

            # Store the stump
            self.stumps.append(stump)

            # Calculate AdaBoost error (optional)
            ada_error = np.sqrt(err * (1 - err))
            self.ada_errors.append(ada_error)

            # Calculate training error
            agg_preds = self.predict(X)
            training_error = np.mean(agg_preds != y)
            self.training_errors.append(training_error)

            print(f"Iteration {t+1}/{iters}, Stump Weight: {stump_weight:.4f}, Error: {err:.4f}, Training Error: {training_error:.4f}")

        return self

    def predict(self, X):
        if not self.stumps:
            raise ValueError("No stumps trained. Call fit() first.")
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])  # Shape: (n_estimators, n_samples)
        stump_preds_binary = np.where(stump_preds == 1, 1, -1)
        agg_preds = np.dot(self.stump_weights, stump_preds_binary)  # Shape: (n_samples,)
        return np.where(agg_preds >= 0, 1, 0)
  
class MultiLabelAdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.classifiers = []
    
    def fit(self, X, Y):
        """
        Train one AdaBoost classifier per class.
        
        Parameters:
        - X: Features, shape (n_samples, n_features)
        - Y: Binary indicator matrix, shape (n_samples, n_classes)
        """
        n_classes = Y.shape[1]
        for i in range(n_classes):
            print(f"Training AdaBoost for class {i+1}/{n_classes}")
            try:
              clf = AdaBoost()
              clf.fit(X, Y[:, i], iters=self.n_estimators)
              self.classifiers.append(clf)
            except Exception as e:
              print("ERROR:")
              print(f"X:{X}")
              print(f"Y:{Y}")
              print(f"clf:{clf}")
              print(f"class:{i}")
              print(e)
              
        return self
    
    def predict(self, X):
        """
        Predict multi-label outputs.
        
        Parameters:
        - X: Features, shape (n_samples, n_features)
        
        Returns:
        - predictions: Binary indicator matrix, shape (n_samples, n_classes)
        """
        predictions = []
        for idx, clf in enumerate(self.classifiers):
            pred = clf.predict(X)
            print(pred)
            # Convert -1/1 to 0/1
            pred_binary = (pred > 0).astype(int)
            predictions.append(pred_binary)
        # print(np.array(predictions).T)
        return np.array(predictions).T  # Shape: (n_samples, n_classes)
