import numpy as np
import matplotlib.pyplot as plt
from KMeans import kMeans
from data import X_train

X_train = np.asarray(X_train, dtype=float)

def distortions(X, centroids):
    distortion = 0
    for i in X:
        distances = np.linalg.norm(centroids - i, axis =1)
        minimum = np.min(distances)
        distortion += minimum**2
    return distortion

k = list(range(1,15))
dis =[]

for i in k:
    model = kMeans(k=i)
    model.train(X_train)
    dis.append(distortions(X_train, model.centroids))

plt.plot(k, dis)
plt.xlabel('k')
plt.ylabel('Distortion')
plt.grid(True)
plt.show()
