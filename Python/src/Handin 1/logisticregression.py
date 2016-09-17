import numpy as np
import sys
import matplotlib.pyplot as plt
#from astropy._erfa.core import cal2jd
#from gevent.ares import result
#from mpmath.tests.test_linalg import improve_solution
class TestResult(object):
    function = ""
    theta = []
    cost_series = []
    error = 0

batch_results = TestResult()
mini_batch_results = TestResult()

def load_data(filename):
    """Loads the training data from file"""
    # Change this to the AU digits data set when it has been released!
    train_file = np.load(filename)
    images = train_file['digits'] #image
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

def load_test_data_27_only(filename):
    return load_data_27_only(filename)

def load_train_data_27_only(filename):
    return load_data_27_only(filename)


def logistic(z):
    """ 
    Computes the logistic function to each entry in input vector z.

    Args:
        z: A vector (numpy array) 
    Returns:
        A new vector of the same length where each entry is the value of the 
        corresponding input entry after applying the logistic function to it
    """
    result = 1 / (1 + np.exp(-z))
    result[result == 1] =  1 - 0.00000000000000009 #WTF
    result[result == 0] = 0.00000000000000009 #WTF
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

    #TODO Jonas: Use filtering to optimize. In the below all calculations are done twice
    linear_terms = np.dot(X, w)
    logistic_terms = logistic(linear_terms)
    log_of_logistic = np.log(logistic_terms)
    positive_errors = np.multiply(log_of_logistic, y)
    log_of_one_minus_logistic = np.log(1 - logistic_terms)
    negative_errors = np.multiply(log_of_one_minus_logistic, (1 - y))
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
    # Ensure arrays are 2d
    cost_series = []
    global batch_results
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)

    if w is None:
        w = np.zeros(X.shape[1]).reshape(X.shape[1],1)
    minimum_improvement = 0.0000001
    speed = 0.0001
    previous_cost = 1000000 #something very high
    keeponimproving = True
    print("Finding the best vector")
    while keeponimproving:
        cost, grad = log_cost(X, y, w)
        improvement = previous_cost - cost
        w = w -(speed*grad)
        batch_results.cost_series.append(cost)
        keeponimproving = improvement < 0 or improvement > minimum_improvement
        previous_cost = cost
    return w

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

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
    global mini_batch_results
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)
    if w is None:
        w = np.zeros(X.shape[1]).reshape(X.shape[1], 1)
    speed = 0.0001
    print("Finding the best vector")
    for x in range(epochs):
        shuffle_in_unison_inplace(X, y)
        i = 0
        while i + batchsize < X.shape[0]:
            cost, grad = log_cost(X, y, w)
            mini_batch_results.cost_series.append(cost)
            print(".", end="")
            w = w - (speed * grad)
            i = i + batchsize
    print("\n")
    return w

def execute_LGMobel(w, x, limit):
    dotproduct = np.dot(x, w)
    logi = logistic(np.array([dotproduct]))
    return 1 if logi > limit else 0



def calcuate_from_model(w, test_img27):
    return np.apply_along_axis(lambda x: execute_LGMobel(w, x, 0.5), 1, test_img27)

def calculate_error(test_img27, test_lab27, w):
    calc_labels = calcuate_from_model(w, test_img27)
    correct_predictions = (calc_labels == test_lab27)
    error = 1 - np.mean(correct_predictions)
    return error

def test_batch_gradient_decent():
    global mini_batch_results
    mini_batch_results.function = "mini_batch with au data, 1 epoch, k = 10"

    train_img27, train_lab27 = load_train_data_27_only('auTrain.npz')
    test_img27, test_lab27 = load_test_data_27_only('auTest.npz')
    w = batch_grad_descent(train_img27, train_lab27)

    batch_results.error = calculate_error(test_img27, test_lab27, w)
    batch_results.theta = w

def test_mini_batch_gradient_decent():
    global mini_batch_results
    mini_batch_results.function = "mini_batch with au data, 1 epoch, k = 10"

    train_img27, train_lab27 = load_train_data_27_only('auTrain.npz')
    test_img27, test_lab27 = load_test_data_27_only('auTest.npz')
    w = mini_batch_grad_descent(train_img27, train_lab27, epochs=1)

    mini_batch_results.error = calculate_error(test_img27, test_lab27, w)
    mini_batch_results.theta = w




def main():
    test_batch_gradient_decent()
    test_mini_batch_gradient_decent()



if __name__ == '__main__':
    main()

