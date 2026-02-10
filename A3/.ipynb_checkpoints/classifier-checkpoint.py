
import numpy as np
from KMeans import kMeans

class Classifier:
    def __init__(self, k, max_iter=100):
        self.kmeans = kMeans(k, max_iter=max_iter)
        self.label = None

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.kmeans.train(X)
        clusters = self.kmeans.predict(X)
        self.label = np.array([
            np.bincount(y[clusters == i].astype(int)).argmax()
            for i in range(self.kmeans.k)
        ])
        return self

    def predict(self, X):
        clusters = self.kmeans.predict(X)
        return self.label[clusters]

    def score(self, X, y):
        # Accuracy Score
        y = np.asarray(y)
        pred = self.predict(X)
        correct = np.sum(y==pred)
        total= len(y)
        accuracy = correct/total
        # F1 Score, MAXXXX VERSSATPPEEEN
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


        return accuracy, F1, precision, recall