import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import math

#Computes for each data point the closest center of a cluster
def closest(data, centers):
    n, d = data.shape
    k, d_ = centers.shape
    assert d == d_

    # Insert your code here
    rep = ...

    # rep should contain a representative index for each data point
    assert rep.shape == (n,)
    assert np.all((0 <= rep) & (rep < k))
    return rep



def kmeans_cost(data, centers, rep):
    n, d = data.shape
    k, d_ = centers.shape
    assert d == d_
    assert rep.shape == (n,)

    # Insert your code here
    data_rep = centers[rep]
    cost = ...

    return cos

def kmeans(data, k, epsilon):
    data = np.asarray(data)#What is the expected input? Is this necessary?
    n, d = data.shape
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    centers = np.array([[np.random.uniform(min[d], max[d]) for d in range(d)] for i in range(k)])

    tired = False
    old_centers = np.zeros_like(centers)
    while not tired:
        old_centers[:] = centers
        clusters = [np.array() for i in range(k)]

        for point in data:
            min_norm = math.inf
            closest_center_index = None
            for center_index in range(len(centers)):
                norm = np.linalg.norm(point - centers[center_index])
                if norm < min_norm:
                    min_norm = norm
                    closest_center_index = center_index
            clusters[closest_center_index].append(point)#This is slow. Find a better solution




        dist = np.sqrt(((centers - old_centers) ** 2).sum(axis=1))
        tired = np.max(dist) <= epsilon

    return centers