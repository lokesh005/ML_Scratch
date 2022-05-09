import numpy as np


def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])


if __name__ == '__main__':
    points = np.vstack(
        (
            (np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
            (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
            (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))
        )
    )

    centroids = initialize_centroids(points, 3)
    closest = closest_centroid(points, centroids)
    new_centroids = move_centroids(points, closest, centroids)
