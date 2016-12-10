import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import sklearn.datasets
from src.Handin4.kmeans import kmeans

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


def main():
    plt.interactive(False)
    iris = sklearn.datasets.load_iris()
    data = iris['data']
    labels = iris['target']
    pca = sklearn.decomposition.PCA(2)
    data_pca = pca.fit_transform(data)

    estimated_labels, centers = kmeans(data_pca, 3, 0.001)
    plot_groups(data_pca, estimated_labels, {0: 'o', 1: 's', 2: '^'}, figsize=(4, 4))
    plt.show()

main()