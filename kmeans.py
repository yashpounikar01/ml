import numpy as np
import matplotlib.pyplot as plt

# K-Means algorithm (concise version)
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k, self.max_iters = k, max_iters

    def fit(self, data):
        self.centroids = data[np.random.choice(len(data), self.k, replace=False)]
        for _ in range(self.max_iters):
            labels = [np.argmin([np.linalg.norm(x - c) for c in self.centroids]) for x in data]
            new_centroids = [np.mean(data[np.array(labels) == i], axis=0) for i in range(self.k)]
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def print_centroids(self):
        print("Final centroids:")
        for i, centroid in enumerate(self.centroids):
            print(f"Centroid {i+1}: {centroid}")

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    data = np.vstack([np.random.normal([0, 0], 0.5, (50, 2)),
                      np.random.normal([3, 3], 0.5, (50, 2)),
                      np.random.normal([0, 3], 0.5, (50, 2))])

    kmeans = KMeans(k=3)
    kmeans.fit(data)
    kmeans.print_centroids()

    # Plot clusters and centroids
    plt.scatter(data[:, 0], data[:, 1], c=[np.argmin([np.linalg.norm(x - c) for c in kmeans.centroids]) for x in data])
    plt.scatter(np.array(kmeans.centroids)[:, 0], np.array(kmeans.centroids)[:, 1], marker='x', s=200, c='red')
    plt.title('K-Means Clustering')
    plt.show()
