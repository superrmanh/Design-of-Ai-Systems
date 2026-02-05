"""
kMeans
"""
import numpy as np
class kMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    # Initialize
    def Initialize(self, X):
        np.random.seed(0)
        random = np.random.permutation(X.shape[0])
        centroid = X[random[:self.k]]
        return centroid
    # Distance
    def Assign(self, X,centroid):
        clusters = []
        for x in X:
            Distance = np.linalg.norm(centroid - x, axis=1)
            cluster = np.argmin(Distance)
            clusters.append(cluster)
        return np.array(clusters)

    # Iterations
    def Update(self, X, clusters):
        new_centroids = []
        for i in range(self.k):
            points = X[clusters == i]
            mean = np.mean(points, axis=0)
            new_centroids.append(mean)
        return np.array(new_centroids)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        centroids = self.Initialize(X)

        for _ in range(self.max_iter):
            clusters = self.Assign(X, centroids)
            centroids = self.Update(X, clusters)

        self.centroids = centroids
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.Assign(X, self.centroids)
    

        
