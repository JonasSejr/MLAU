import numpy as np
import matplotlib.pyplot as plt
from astropy._erfa.core import cal2jd
from mpmath.tests.test_linalg import improve_solution


def load_data(filename):
    """Loads the training data from file"""
    # Change this to the AU digits data set when it has been released!
    train_file = np.load(filename)
    images = train_file['images']
    labels = np.squeeze(train_file['labels'])  # TODO remove squeeze
    print('Shape of input data: %s %s' % (images.shape, labels.shape))
    return images, labels

def load_data_27_only(filename):
    digits, labels = load_data(filename)
    idx1 = (labels == 2)
    idx2 = (labels == 7)
    print(idx1.shape)
    img27 = digits[idx1 | idx2,:]
    lab27 = labels[idx1 | idx2]
    lab27[lab27==2] = 0
    lab27[lab27==7] = 1
    return img27, lab27

def load_test_data_27_only():
    return load_data_27_only('mnistTest.npz')

def load_train_data_27_only():
    return load_data_27_only('mnistTrain.npz')


def logistic(z):
    """ 
    Computes the logistic function to each entry in input vector z.

    Args:
        z: A vector (numpy array) 
    Returns:
        A new vector of the same length where each entry is the value of the 
        corresponding input entry after applying the logistic function to it
    """
    return 1/(1 + np.exp(-z))

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

    #Jonas: Use filtering to optimize. In the below all calculations are done twice
    linear_terms = np.dot(X, w)
    logistic_terms = logistic(linear_terms)
    positive_errors = np.multiply(np.log(logistic_terms), y)
    negative_errors = np.multiply(np.log(1 - logistic_terms), (1 - y))
    all_errors = np.concatenate((positive_errors, negative_errors))
    cost = -(np.sum(all_errors)/np.shape(X)[0])

    grad = -np.dot(X.T,(y - logistic_terms))

    return cost, grad

def batch_grad_descent(X, y, w=None, reg=0):
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
    print('Please implement me and use all data in each gradient computation')
    # Ensure arrays are 2d
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)

    if w is None:
        w = np.zeros(X.shape[1]).reshape(X.shape[1],1)
    minimum_improvement = 0.00000000001
    speed = 100
    previous_cost = 1000000 #something very high
    keeponimproving = True
    while keeponimproving:
        cost, grad = log_cost(X, y, w)
        improvement = previous_cost - cost
        w = w + speed*grad
        print(w)
        keeponimproving = improvement < 0 or improvement > minimum_improvement
    return w

def mini_batch_grad_descent(X, y, w=None, reg=0, batchsize=16, epochs=10):
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
    print('Please implement me and use "batchsize" data points in each gradient computation')
    if w is None:
        w = np.zeros(X.shape[1])
    n = X.shape[0]
    return w

def execute_LGMobel(w, x, limit):
    dotproduct = np.dot(x, w)
    logi = logistic(dotproduct)
    return 1 if logi > limit else 0

def main():
    """An example main method. You should replace this with whatever you need."""
    train_img27, train_lab27 = load_train_data_27_only()
    test_img27, test_lab27 = load_test_data_27_only()
    w = batch_grad_descent(train_img27, train_lab27)
    calc_labels = np.apply_along_axis(lambda x: execute_LGMobel(w, x, 0.5), 1, test_img27)
    errors = (calc_labels == test_lab27)
    print("Percent positive = " + str(np.mean(test_lab27)))
    print("Error ration = " + str(np.mean(errors)))
    #plt.imshow(w.reshape(28, -1, order='F'), cmap='bone')
    #print('Look look it is a pretty two i think')

if __name__ == '__main__':
    main()

