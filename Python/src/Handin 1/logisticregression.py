import numpy as np
import matplotlib.pyplot as plt
import time


def load_data(filename):
    """Loads the training data from file"""
    # Change this to the AU digits data set when it has been released!
    train_file = np.load(filename)
    images = train_file['digits']  # image
    labels = np.squeeze(train_file['labels'])  # TODO remove squeeze
    return images, labels


def load_data_mnist(filename):
    """Loads the training data from file"""
    # Change this to the AU digits data set when it has been released!
    train_file = np.load(filename)
    images = train_file['images']  # image
    labels = np.squeeze(train_file['labels'])  # TODO remove squeeze
    return images, labels

def load_data_27_only(filename):
    digits, labels = load_data(filename)
    idx1 = (labels == 2)
    idx2 = (labels == 7)
    img27 = digits[idx1 | idx2, :]
    lab27 = labels[idx1 | idx2]
    lab27[lab27 == 2] = 0
    lab27[lab27 == 7] = 1
    return img27, lab27

def load_data_27_only_mnist(filename):
    digits, labels = load_data_mnist(filename)
    idx1 = (labels == 2)
    idx2 = (labels == 7)
    img27 = digits[idx1 | idx2, :]
    lab27 = labels[idx1 | idx2]
    lab27[lab27 == 2] = 0
    lab27[lab27 == 7] = 1
    return img27, lab27


def logistic(z):
    """ 
    Computes the logistic function to each entry in input vector z.

    Args:
        z: A vector (numpy array) 
    Returns:
        A new vector of the same length where each entry is the value of the 
        corresponding input entry after applying the logistic function to it
    """
    exp = np.exp(-z)
    result = 1 / (1 + exp)
    very_close_to_1 = 1 - 0.00000000000000009
    very_close_to_0 = 0.00000000000000009
    result[result > very_close_to_1] = very_close_to_1  # WTF
    result[result < very_close_to_0] = very_close_to_0  # WTF
    return result


def log_cost(X, y, w, reg=0):
    """
    Compute the regularized cross entropy and the gradient under the logistic regression model 
    using data X,y and weight vector w

    Args:
        X: A 2D numpy array of data points stored row-wise (n x d)
        y: A 1D numpy array of target values (n x 1)
        w: A 1D numpy array of weights (d x 1)
        reg: Optional regularization parameter
    Returns:
        cost: Average Negative Log Likelihood of w
        grad: The gradient of the average Negative Log Likelihood at w
    """

    # TODO Jonas: Use filtering to optimize. In the below all calculations are done twice
    linear_terms = np.dot(X, w)
    logistic_terms = logistic(linear_terms)
    log_of_logistic = np.log(logistic_terms)
    positive_errors = np.multiply(log_of_logistic, y)
    log_of_one_minus_logistic = np.log(1 - logistic_terms)
    negative_errors = np.multiply(log_of_one_minus_logistic, (1 - y))
    all_errors = np.concatenate((positive_errors, negative_errors))
    cost = -(np.sum(all_errors) / np.shape(X)[0]) + reg*np.sum(np.dot(w.T, w))

    grad = -np.dot(X.T, (y - logistic_terms)) + reg*2*w

    return cost, grad


def batch_grad_descent(X, y, w=None, epochs=float('inf'), reg=0, speed=0.1, seconds=float('inf')):
    """ 
    Run Batch Gradient Descent on data X,y to minimize the NLL for logistic regression on data X,y

    Args:
        X: A 2D numpy array of data points stored row-wise (n x d)
        y: A 1D numpy array of target values (n x 1)
        w: A 1D numpy array of weights (d x 1)
        reg: Optional regularization parameter
    Returns:
        Learned weight vector (d x 1)
    """
    # Ensure arrays are 2d
    cost_series = []
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)

    if w is None:
        w = np.zeros(X.shape[1]).reshape(X.shape[1], 1)
    keeponimproving = True
    iteration = 0
    start_time = time.time()

    while keeponimproving:
        iteration = iteration + 1
        cost, grad = log_cost(X, y, w, reg=reg)
        w = w - (speed * grad)
        cost_series.append(cost)
        time_taken = time.time() - start_time
        keeponimproving = iteration < epochs and time_taken < seconds
    total_time = time.time() - start_time
    return w, cost_series, total_time


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def mini_batch_grad_descent(X, y, w=None, reg=0, batchsize=16, epochs=10, speed=0.1, seconds=float('inf')):
    """
    Run Mini-Batch Gradient Descent on data X,y to minimize the NLL for logistic regression on data X,y
    The input defines mini-batch size and the number of passess through the data set (epochs)

    Args:
        X: A 2D numpy array of data points stored row-wise (n x d)
        y: A 1D numpy array of target values (n x 1)
        w: A 1D numpy array of weights (d x 1)
        reg: Optional regularization parameter
        batchsize: Number of data points to use in each batch
        epochs: Number of times to go over all data points
    Returns:
        Learned weight vector (d x 1)
    """
    cost_series = []
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)
    if w is None:
        w = np.zeros(X.shape[1]).reshape(X.shape[1], 1)
    start_time = time.time()
    for x in range(epochs):
        shuffle_in_unison_inplace(X, y)
        i = 0
        while i + batchsize < X.shape[0]:
            cost, grad = log_cost(X[i:i + batchsize,:], y[i:i + batchsize,:], w)
            cost_series.append(cost)
            w = w - (speed * grad)
            i = i + batchsize
            if(time.time() - start_time > seconds):
                break
        if(time.time() - start_time > seconds):
            break
        total_time = time.time() - start_time
    return w, cost_series, total_time


def all_versus_one_lg_mini_batch(X, y, w=None, reg=0, batchsize=10, epochs=10, speed=0.1):
    models = []
    min_value = min(y)
    max_value = max(y)
    # We assume all values exist.
    for i in range(min_value, max_value + 1):
        y_bin = np.copy(y)
        index_i = (y_bin == i)
        index_not_i = (y_bin != i)
        y_bin[index_i] = 1
        y_bin[index_not_i] = 0
        model, cost_series, total_time = mini_batch_grad_descent(X, y_bin, w=w, batchsize=batchsize, reg=reg, epochs=epochs, speed=speed)
        models.append(model)
        print("model trained")
    return models;


def get_probability_lg_model(w, x):
    dotproduct = np.dot(x, w)
    probability = logistic(np.array([dotproduct]))
    return probability


def get_class_lg_model(w, x, limit):
    probability = get_probability_lg_model(w, x)
    return 1 if probability > limit else 0


def get_class_multiclass_lg_model(models, x):
    probabilities = []
    for w in models:
        probability = get_probability_lg_model(w, x)
        probabilities.append(probability)
    return probabilities.index(max(probabilities))


def calculate_labels_with_model(w, X):
    return np.apply_along_axis(lambda x: get_class_lg_model(w, x, 0.5), 1, X)


def calculate_error(X, y, w):
    calc_labels = calculate_labels_with_model(w, X)
    misclassifications = np.not_equal(calc_labels.reshape((calc_labels.size, 1)), y.reshape((y.size, 1)))
    error = np.mean(misclassifications)
    return error, X[misclassifications.flatten()]


def calculate_multiclass_labels_with_model(models, X):
    return np.apply_along_axis(lambda x: get_class_multiclass_lg_model(x=x, models=models), 1, X)


def calculate_multiclass_error(X, y, models):
    calc_labels = calculate_multiclass_labels_with_model(models=models, X=X)
    misclassifications = np.not_equal(calc_labels.reshape((calc_labels.size, 1)), y.reshape((y.size, 1)))
    error = np.mean(misclassifications)
    return error, X[misclassifications.flatten()]


def test_7_v_2_batch(reg, speed, epochs=float('inf'), seconds=float('inf')):
    train_img, train_lab = load_data_27_only('auTrain.npz')
    test_img, test_lab = load_data_27_only('auTest.npz')
    w, cost_series, total_time = batch_grad_descent(X=train_img, y=train_lab, reg=reg, epochs=epochs, speed=speed,
                                                    seconds=seconds)
    error, misclassified_images = calculate_error(X=test_img, y=test_lab, w=w)
    return w, error, cost_series, misclassified_images, total_time

def test_7_v_2_batch_MNIST(reg, speed, epochs=float('inf'), seconds=float('inf')):
    train_img, train_lab = load_data_27_only_mnist('mnistTrain.npz')
    test_img, test_lab = load_data_27_only_mnist('mnistTest.npz')
    w, cost_series, total_time = batch_grad_descent(X=train_img, y=train_lab, reg=reg, epochs=epochs, speed=speed,
                                                    seconds=seconds)
    error, misclassified_images = calculate_error(X=test_img, y=test_lab, w=w)
    return w, error, cost_series, misclassified_images, total_time

def test_7_v_2_mini_batch(reg, speed, batchsize, epochs=float('inf'), seconds=float('inf')):
    train_img, train_lab = load_data_27_only('auTrain.npz')
    test_img, test_lab = load_data_27_only('auTest.npz')
    w, cost_series, total_time = mini_batch_grad_descent(X=train_img, y=train_lab, reg=reg, batchsize=batchsize,
                                                         epochs=epochs, speed=speed, seconds=seconds)
    error, misclassified_images = calculate_error(X=test_img, y=test_lab, w=w)
    return w, error, cost_series, misclassified_images, total_time


def test_all_v_one(speed, reg=0, batchsize=16, epochs=10):
    train_img, train_lab = load_data('auTrain.npz')
    test_img, test_lab = load_data('auTest.npz')
    models = all_versus_one_lg_mini_batch(X=train_img, y=train_lab, reg=reg, batchsize=batchsize, epochs=epochs, speed=speed)
    error, misclassified_images = calculate_multiclass_error(X=test_img, y=test_lab, models=models)
    return models, error, misclassified_images


def show_images(images):
    x = images.reshape(-1, 28, 28)
    x = x.transpose(1, 0, 2)
    plt.imshow(x.reshape(28, -1), cmap='bone')
    plt.yticks([])
    plt.xticks([])
    plt.figure()


def analyseSimpleBatchGradientDescentMNIST():
    w_batch, error_batch, cost_series_batch, misclassified_images_batch, total_time_batch = \
        test_7_v_2_batch_MNIST(reg=0, epochs=1000, speed=0.05)
    print("Error " + str(error_batch))

def analyseSimpleBatchGradientDescent():
    w_batch, error_batch, cost_series_batch, misclassified_images_batch, total_time_batch = \
        test_7_v_2_batch(reg=0, epochs=1000, speed=0.05)

    print("Total time" + str(total_time_batch))
    np.savez("params.npz", theta=w_batch)
    print("Error " + str(error_batch))
    print("Number of misclassifications " + str(misclassified_images_batch.shape[0]) + " of " + "?")

    plt.imshow(w_batch.reshape(28, 28), cmap='bone')
    plt.show()
    show_images(misclassified_images_batch[0:16, :])

    plt.plot(cost_series_batch, 'r', linewidth=2)
    plt.title('cost per iteration')
    plt.show()

def analyseSimpleBatchGradientDescentRegularization():
    w_batch, error_batch, cost_series_batch, misclassified_images_batch, total_time_batch = \
        test_7_v_2_batch(reg=0.3, epochs=1000, speed=0.05)

    print("Total time" + str(total_time_batch))
    np.savez("params.npz", theta=w_batch)
    print("Error " + str(error_batch))
    print("Number of misclassifications " + str(misclassified_images_batch.shape[0]) + " of " + "?")

    plt.imshow(w_batch.reshape(28, 28), cmap='bone')
    plt.show()
    show_images(misclassified_images_batch[0:16, :])

    plt.plot(cost_series_batch, 'r', linewidth=2)
    plt.title('cost per iteration')
    plt.show()

def analyseBatchVersusMiniBatch():
    w_batch, error_batch, cost_series_batch, misclassified_images_batch, total_time_batch = \
        test_7_v_2_batch(seconds=10, reg=0, epochs=1000, speed=0.05)
    w_mini_batch, error_mini_batch, cost_series_mini_batch, misclassified_images_mini_batch, total_time_mini_batch = \
        test_7_v_2_mini_batch(seconds=10, reg=0, epochs=1000, speed=0.05, batchsize=10)
    print("Error batch " + str(error_batch))
    print("Error mini batch " + str(error_mini_batch))

    plt.plot(cost_series_batch, 'r', linewidth=2)
    plt.title('cost per iteration (batch)')
    plt.show()

    plt.plot(cost_series_mini_batch, 'r', linewidth=2)
    plt.title('cost per iteration (mini_batch)')
    plt.show()

def analyseOneVersusAll():
    models, error, misclassified_images = \
        test_all_v_one(reg=0, epochs=1, speed=0.05, batchsize=100)

    print("Error " + str(error))
    print("Number of misclassifications " + str(misclassified_images.shape[0]) + " of " + "?")
    flattened = []
    for w in models:
        flattened.append(w.flatten())
    show_images(np.array(flattened))
    plt.show()
    show_images(misclassified_images[0:10, :])
    plt.show()


def main():
    plt.interactive(False)
    #analyseSimpleBatchGradientDescent()
    #analyseSimpleBatchGradientDescentRegularization()
    #analyseSimpleBatchGradientDescentMNIST()
    analyseBatchVersusMiniBatch()
    #analyseOneVersusAll()
    # TODO[Jonas]: Bonus: regularization
    # TODO[Jonas]: Bonus: Test algs mnist
    return


if __name__ == '__main__':
    main()
