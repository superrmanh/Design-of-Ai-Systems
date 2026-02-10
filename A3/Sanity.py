# Testings and Sanity Check
from sklearn.datasets import make_blobs
from KMeans import kMeans
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)


X_synthetic, y_true = make_blobs(n_samples=300, n_features=2, centers=4, 
                                 cluster_std=0.8, random_state=42)



model = kMeans(k=4, max_iter=100)
model.train(X_synthetic)
labels = model.predict(X_synthetic)
centroids = model.centroids

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=y_true, cmap='viridis', alpha=0.6)
axes[0].set_title('True clusters (synthetic)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

axes[1].scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=labels, cmap='viridis', alpha=0.6)
axes[1].scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', edgecolors='black', linewidths=2, label='Centroids')
axes[1].set_title('KMeans clusters')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()

plt.tight_layout()
plt.show()