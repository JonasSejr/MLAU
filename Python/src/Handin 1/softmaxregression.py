import numpy as np
import matplotlib.pyplot as plt
import time


def softmax(X):
    """
    Compute the softmax of each row of an input matrix (2D numpy array).
    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option.
    Remember to handle the numerical problems as discussed above.

    Args:
        X: A 2D numpy array
    Returns:
        A new 2D numpy array of the same size as the input where each row in the input
        has been replaced by the softmax of that row
    """
    maxs = np.max(X, axis=1, keepdims=True)
    Xnorm = X - maxs
    exp_1 = np.exp(Xnorm)
    sums = np.sum(exp_1, axis=1, keepdims=True)
    exp_2 = X - maxs - np.log(sums)
    result = np.exp(exp_2)
    very_close_to_0 = 0.00000000000000009
    result[result < very_close_to_0] = very_close_to_0  # WTF
    return result


def soft_cost(X, Y, W, reg=0):
    """
    Compute the regularized cross entropy and the gradient under the logistic regression model
    using data X,y and weight vector w.

    Args:
        X: A 2D numpy array of data points stored row-wise (n x d)
        Y: A 2D numpy array of target values in 1-in-K encoding (n x K)
        W: A 2D numpy array of weights (d x K)
        reg: Optional regularization parameter
    Returns:
        totalcost: Average Negative Log Likelihood of w
        gradient: The gradient of the average Negative Log Likelihood at w
    """
    linearPart = np.dot(X, W)
    softmaxPart = softmax(linearPart)
    logPart = np.log(softmaxPart)
    NLLs = np.multiply(Y,logPart)
    #We need the rowwise dotproduct, but we get elementwise product.
    #In the next line we sum all the elements and therefore we end up with the same result
    totalcost = -(np.sum(NLLs))/X.shape[0]
    gradient = -np.dot(X.T, (Y - softmax(np.dot(X, W))))
    return totalcost, gradient


def batch_grad_descent(X, Y, W=None, max_iterations=float('inf'), reg=0.0, speed=0.1, seconds=float('inf')):
    """
    Run batch gradient descent to learn softmax regression weights that minimize the in-sample error
    Args:
        X: A 2D numpy array of data points stored row-wise (n x d)
        Y: A 2D numpy array of target values in 1-in-K encoding (n x K)
        W: A 2D numpy array of weights (d x K)
        reg: Optional regularization parameter
    Returns:
        The learned weights
    """
    cost_series = []
    if W is None:
        W = np.zeros((X.shape[1], Y.shape[1]))
    print("Finding the best vector")
    iteration = 0
    keeponimproving = True
    start_time = time.time()
    while keeponimproving:
        iteration += 1
        cost, grad = soft_cost(X, Y, W)
        cost_series.append(cost)
        W -= speed * grad
        time_taken = time.time() - start_time
        keeponimproving = iteration < max_iterations and time_taken < seconds
        print(cost)
    total_time = time.time() - start_time
    return W, cost_series, total_time

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def mini_batch_grad_descent(X, Y, W=None, batchsize=16, epochs=100, speed=0.1, seconds=float('inf')):
    """
    Run Mini-Batch Gradient Descent on data X,y to minimize the NLL for logistic regression on data X,y
    Args:
        X: A 2D numpy array of data points stored row-wise (n x d)
        Y: A 2D numpy array of target values in 1-in-K encoding (n x K)
        W: A 2D numpy array of weights (d x K)
        reg: Optional regularization parameter
        batchsize: size of mini-batch
        epochs: Number of iterations through the data to use
    Returns:
        The learned weights
    """
    cost_series = []
    if W is None:
        W = np.zeros((X.shape[1], Y.shape[1]))
    start_time = time.time()
    for x in range(epochs):
        shuffle_in_unison_inplace(X, Y)
        i = 0
        while i + batchsize < X.shape[0]:
            cost, grad = soft_cost(X[i:i+batchsize, :], Y[i:i+batchsize, :], W)
            cost_series.append(cost)
            W -= speed * grad
            i = i + batchsize
            if (time.time() - start_time > seconds):
                break
        if (time.time() - start_time > seconds):
            break
        total_time = time.time() - start_time
        print(cost)
    return W, cost_series, total_time


def load_data(filename):
    """Loads the training data from file"""
    # Change this to the AU digits data set when it has been released!
    train_file = np.load(filename)
    images = train_file['digits']  # image
    labels = np.squeeze(train_file['labels'])  # TODO remove squeeze
    print('Shape of input data: %s %s' % (images.shape, labels.shape))
    return images, labels

def load_data_mnist(filename):
    """Loads the training data from file"""
    # Change this to the AU digits data set when it has been released!
    train_file = np.load(filename)
    images = train_file['images']  # image
    labels = np.squeeze(train_file['labels'])  # TODO remove squeeze
    print('Shape of input data: %s %s' % (images.shape, labels.shape))
    return images, labels


def convertLabels(labels):
    one_hot = np.zeros((labels.size, np.max(labels) + 1))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def calculate_labels_with_softmax_model(W, X):
    return np.argmax(softmax(np.dot(X, W)), axis=1)

def calculate_multiclass_error(X, y, W):
    calc_labels = calculate_labels_with_softmax_model(W=W, X=X)
    misclassifications = np.not_equal(calc_labels.reshape((calc_labels.size, 1)), y.reshape((y.size, 1)))
    error = np.mean(misclassifications)
    return error, X[misclassifications.flatten()]

def show_images(images):
    x = images.reshape(-1, 28, 28)
    x = x.transpose(1, 0, 2)
    plt.imshow(x.reshape(28, -1), cmap='bone')
    plt.yticks([])
    plt.xticks([])
    plt.figure()

def analyseSimpleBatchGradientDescent():
    train_img, train_lab = load_data('auTrain.npz')
    test_img, test_lab = load_data('auTrain.npz')
    softmax_train_lab = convertLabels(train_lab)
    W, cost_series, total_time = batch_grad_descent(train_img, softmax_train_lab, max_iterations=1000, speed= 0.00002)
    error, misclassified_images_batch = calculate_multiclass_error(test_img, test_lab, W)
    print("Error " + str(error))
    show_images(misclassified_images_batch[0:16, :])
    plt.show()
    show_images(W.T)
    plt.show()
    plt.plot(cost_series, 'r', linewidth=2)
    plt.title('cost per iteration softmax batch')
    plt.show()
    np.savez("paramsSoftmax.npz", theta=W)

def analyseSimpleBatchGradientDescentMNIST():
    train_img, train_lab = load_data_mnist('mnistTrain.npz')
    test_img, test_lab = load_data_mnist('mnistTrain.npz')
    softmax_train_lab = convertLabels(train_lab)
    W, cost_series, total_time = batch_grad_descent(train_img, softmax_train_lab, max_iterations=1000, speed= 0.00002)
    error, misclassified_images_batch = calculate_multiclass_error(test_img, test_lab, W)
    print("Error " + str(error))

def analyseMiniBatchVersusBatchGradienDescent():
    train_img, train_lab = load_data('auTrain.npz')
    test_img, test_lab = load_data('auTrain.npz')
    softmax_train_lab = convertLabels(train_lab)
    W, cost_series, total_time = batch_grad_descent(train_img, softmax_train_lab, seconds=60,speed= 0.00002)
    error, misclassified_images_batch = calculate_multiclass_error(test_img, test_lab, W)
    W_mini, cost_series_mini, total_time_mini = mini_batch_grad_descent(train_img, softmax_train_lab, epochs=100000, seconds=60, speed=0.00002)
    error_mini, misclassified_images_batch_mini = calculate_multiclass_error(test_img, test_lab, W_mini)
    print("Batch " + str(error))
    print("Mini Batch " + str(error_mini))

    plt.plot(cost_series, 'r', linewidth=2)
    plt.title('cost per iteration softmax batch')
    plt.show()

    plt.plot(cost_series_mini, 'r', linewidth=2)
    plt.title('cost per iteration softmax mini batch')
    plt.show()

if __name__ == "__main__":
    plt.interactive(False)
    #analyseSimpleBatchGradientDescent()
    #analyseSimpleBatchGradientDescentMNIST()
    analyseMiniBatchVersusBatchGradienDescent()