from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np
import sklearn.decomposition
import sklearn.datasets



def plot_matrix(x, y, group, fmt='.', **kwargs):
    """
    Given two d-dimensional datasets of n points,
    makes a figure containing d x d plots, where the (i, j) plot
    plots the ith dimension against the jth dimension.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    group = np.squeeze(np.asarray(group))
    n, p = x.shape
    n_, q = y.shape
    n__, = group.shape
    assert n == n_ == n__
    groups = sorted(set(group))
    if isinstance(fmt, str):
        fmt = {k: fmt for k in groups}
    fig, axes = plt.subplots(p, q, squeeze=False, **kwargs)
    for i, axrow in enumerate(axes):
        for j, ax in enumerate(axrow):
            for g in groups:
                ax.plot(x[group == g, i], y[group == g, j], fmt[g])
            if len(axes) > 2:
                ax.locator_params(tight=True, nbins=4)


def plot_groups(x, group, fmt='.', **kwargs):
    """
    Helper function for plotting a 2-dimensional dataset with groups
    using plot_matrix.
    """
    n, d = x.shape
    assert d == 2
    x1 = x[:, 0].reshape(n, 1)
    x2 = x[:, 1].reshape(n, 1)
    plot_matrix(x1, x2, group, fmt, **kwargs)


# The first function takes a description of a Gaussian Mixture (that is, the mean, covariance matrix and prior of each Gaussian)
# and returns the probability densities of each point.
def pdf(points, mean, cov, prior):
    points, mean, cov = np.asarray(points), np.asarray(mean), np.asarray(cov)
    prior = np.asarray(prior)
    n, d = points.shape
    k, d_1 = mean.shape
    k_2, d_2, d_3 = cov.shape
    k_3, = prior.shape
    assert d == d_1 == d_2 == d_3
    assert k == k_2 == k_3, "%s %s %s should be equal" % (k, k_2, k_3)

    # Compute probabilities
    prob = []
    for i in range(k):
        if prior[i] < 1 / k ** 3:
            prob.append(np.zeros(n))
        else:
            prob.append(
                prior[i] *
                multivariate_normal.pdf(
                    mean=mean[i], cov=cov[i], x=points))
    prob = np.transpose(prob)  # n x k
    # Normalize cluster probabilities of each point
    prob = prob / np.sum(prob, axis=1, keepdims=True)  # n x k

    assert prob.shape == (n, k)
    assert np.allclose(prob.sum(axis=1), 1)
    return prob



# The following helper function computes the most likely class of each point under a given Gaussian Mixture.
def most_likely(points, mean, cov, prior):
    prob = pdf(points, mean, cov, prior)
    return np.argmax(prob, axis=1)

def em(points, k, epsilon, mean=None):
    points = np.asarray(points)
    n, d = points.shape

    # Initialize and validate mean
    if mean is None:
        min = np.min(points, axis=0)
        max = np.max(points, axis=0)
        mean = np.array([[np.random.uniform(min[d], max[d]) for d in range(d)] for i in range(k)])

    # Validate input
    mean = np.asarray(mean)
    k_, d_ = mean.shape
    assert k == k_
    assert d == d_

    # Initialize cov, prior
    cov = [np.identity(d) for i in range(k)]
    prior = [1/k for i in range(k)]

    tired = False
    old_mean = np.zeros_like(mean)
    while not tired:
        old_mean[:] = mean
        weights = pdf(points, mean, cov, prior)
        # Maximization step

        mean = []
        cov = []
        prior = []
        for i in range(k):
            weighted_points = [points[j] * weights[j, i] for j in range(n)]
            weighted_points = np.asarray(weighted_points)

            weight_sum = np.sum(weights[:, i])

            current_weighted_mean = np.sum(weighted_points, axis=0)/weight_sum
            mean.append(np.asarray(current_weighted_mean))

            current_mean = np.mean(points[i], axis=0)
            current_cov = np.cov(points, rowvar=False, aweights=weights[:, i])#/weight_sum
            #[[np.random.uniform(min[d], max[d]) for d in range(d)] for i in range(k)]

            cov.append(np.asarray(current_cov))

            prior.append(weight_sum/n)

        mean = np.asarray(mean)
        cov = np.asarray(cov)
        prior = np.asarray(prior)

        # Finish condition
        dist = np.sqrt(((mean - old_mean) ** 2).sum(axis=1))
        print(dist)
        tired = np.all(dist < epsilon)
        estimated_labels = most_likely(points, mean, cov, prior)
        #print(mean)
        #plot_groups(points, estimated_labels, {0: 'o', 1: 's', 2: '^'}, figsize=(4, 4))
        #plt.show()

    # Validate output
    assert mean.shape == (k, d)
    assert cov.shape == (k, d, d)
    assert prior.shape == (k,)
    return mean, cov, prior