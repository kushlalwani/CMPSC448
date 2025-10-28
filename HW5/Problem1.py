import numpy as np
import scipy as sp


def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    n_samples, d_features = X.shape
    centers = [X[np.random.randint(0,n_samples)]] #randomly choose a first center for the clusters

    for i in range(1,k):
        distances = np.array([min(np.sum((x - c) ** 2) for c in centers) for x in X])
        probabilities = distances/np.sum(distances)
        cum_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for index, prob in enumerate(cum_probabilities):
            if r < prob:
                centers.append(X[index])
                break
    
    return np.array(centers)

def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """
    centers = k_init(X, k)

    for i in range(max_iter):
        data_map = assign_data2clusters(X, centers)
        new_centers = np.zeros_like(centers)

        for j in range(k):
            assigned_points = X[data_map[:,j]==1]
            if len(assigned_points) > 0:
                new_centers[j] = np.mean(assigned_points, axis = 0)
            else:
                new_centers[j] = X[np.random.randint(0, X.shape[0])]

        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return centers


def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    n = X.shape[0]
    k = C.shape[0]
    data_map = np.zeros((n, k))

    for i in range(n):
        distances = np.linalg.norm(X[i] - C, axis = 1)
        cluster = np.argmin(distances)
        data_map[i, cluster] = 1

    return data_map


def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    n = X.shape[0]
    total_distortion = 0.0

    for i in range(n):
        distances = np.linalg.norm(X[i] - C, axis = 1)
        min_dist = np.min(distances)
        total_distortion += min_dist ** 2

    return total_distortion
