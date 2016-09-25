import numpy as np
import matplotlib.pyplot as plt
from blaze.compute.tests.test_numpy_compute import test_label


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

    #Simple implementation
    #Exps = np.exp(X)
    #denominator = np.sum(Exps, axis=1)
    #result = Exps / (denominator.reshape(len(Exps), 1))
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
    NNLs = np.multiply(softmaxPart, Y)
    totalcost = np.sum(NNLs)
    gradient = -np.dot(X.T, (Y - softmax(np.dot(X, W))))
    return totalcost, gradient


def batch_grad_descent(X, Y, W=None, max_iterations=100, reg=0.0):
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
    test_img, test_lab = load_data('auTrain.npz')

    if W is None:
        W = np.zeros((X.shape[1], Y.shape[1]))
    print("Finding the best vector")
    speed = 0.1  # Recommended in the book
    iteration = 0
    keeponimproving = True
    while keeponimproving:
        iteration += 1
        cost, grad = soft_cost(X, Y, W)
        W -= speed * grad
        keeponimproving = iteration < max_iterations
        print(calculate_multiclass_error(test_img, test_lab, W))
    return W


def mini_batch_grad_descent(X, y, reg, W=None, batchsize=16, epochs=10):
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
    if W is None:
        W = np.zeros((X.shape[1], y.shape[1]))
    return W



def load_data(filename):
    """Loads the training data from file"""
    # Change this to the AU digits data set when it has been released!
    train_file = np.load(filename)
    images = train_file['digits']  # image
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
    correct_predictions = (calc_labels == y)
    error = 1 - np.mean(correct_predictions)
    return error

def test_softmax():
    train_img, train_lab = load_data('auTrain.npz')
    test_img, test_lab = load_data('auTrain.npz')
    softmax_train_lab = convertLabels(train_lab)
    W = batch_grad_descent(train_img, softmax_train_lab, max_iterations=100)
    print(calculate_multiclass_error(test_img, test_lab, W))

if __name__ == "__main__":
    test_softmax()
