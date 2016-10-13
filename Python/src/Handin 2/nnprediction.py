import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_data_au(filename):
    train_file = np.load(filename)
    images = train_file['digits']  # image
    labels = np.squeeze(train_file['labels'])  # TODO remove squeeze
    print('Shape of input data: %s %s' % (images.shape, labels.shape))
    return images, labels

def load_data_mnist(filename):
    train_file = np.load(filename)
    images = train_file['images']  # image
    labels = np.squeeze(train_file['labels'])  # TODO remove squeeze
    return images, labels


if __name__ == "__main__":
    #plt.interactive(False)
    mnist_train_img, mnisttrain_lab = load_data('data/auTrain.npz')
    mnisttest_img, mnisttest_lab = load_data('data/auTrain.npz')
    