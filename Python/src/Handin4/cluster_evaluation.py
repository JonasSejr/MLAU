import numpy as np


def calcuate_contingency(labels, predicted):
    cluster_count = np.max(predicted) + 1
    labels_count = np.max(labels) + 1
    contingency = np.empty((cluster_count, labels_count))
    for i in range(cluster_count):
        for j in range(labels_count):
            contingency[i][j] = np.logical_and(np.equal(predicted, i), np.equal(labels, j)).sum()
    return contingency


def calculate_f1(predicted, labels):
    n, = predicted.shape
    assert labels.shape == (n,)
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

    contingency =  calcuate_contingency(labels, predicted)
    precision = np.max(contingency)/np.sum(contingency, axis=1)
    recall = np.empty(r)
    for i in range(r):
        max_index = np.argmax(contingency[i,:])
        recall[i] = contingency[i][max_index]/np.sum(contingency[:, max_index])

    F_individual = 2/(1/precision + 1/recall)
    F_overall = np.mean(F_individual)

    assert contingency.shape == (r, k)
    return F_individual, F_overall, contingency

def calculate_silhouettes(data, predicted):
    data = np.asarray(data)
    n, d = data.shape
    predicted = np.squeeze(np.asarray(predicted))
    k = np.max(predicted) + 1
    assert predicted.shape == (n,)

    distances = np.empty((n, k))
    for i in range(len(data)):
        current_point = data[i]
        data_excluding_current = np.delete(data, i, axis=0)
        predicted_excluding_current = np.delete(predicted, i, axis=0)
        for j in range(k):
            cluster_elements = data_excluding_current[predicted_excluding_current == j]
            p_to_p = np.linalg.norm(cluster_elements - current_point, axis=1)
            distances[i][j]= np.mean(p_to_p)
    silhouettes = np.empty(n)
    for i in range(len(data)):
        current_cluster = predicted[i]
        current_cluster_distances = distances[i]
        dist_in = current_cluster_distances[current_cluster]
        other_cluster_distances = np.delete(current_cluster_distances, current_cluster)
        min_dist_out = np.min(other_cluster_distances)
        point_silhouette = (min_dist_out - dist_in)/max((min_dist_out, dist_in))
        silhouettes[i] = point_silhouette

    assert silhouettes.shape == (n,)
    return silhouettes

