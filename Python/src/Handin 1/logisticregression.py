import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Loads the training data from file"""
    # Change this to the AU digits data set when it has been released!
    train_file = np.load('mnistTrain.npz')
    images = train_file['images']
    labels = np.squeeze(train_file['labels'])  # TODO remove squeeze
    print('Shape of input data: %s %s' % (images.shape, labels.shape))
    return images, labels

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
    #Use filtering
    cost = -(np.sum(y.T*np.log(logistic(w*X.T))) + np.sum((1 - y.T)*np.log(1 - logistic(w*X.T))))
    print('Please implement me without (too many) for-loops. I am important!')
    grad = 0
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
    if w is None:
        w = np.zeros(X.shape[1])
    w = X[0, :]
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

def main():
    """An example main method. You should replace this with whatever you need."""
    print('You rang sir/miss')
    digits, labels = load_data()
    idx1 = (labels == 2)
    idx2 = (labels == 7)
    print(idx1.shape)
    img27 = digits[idx1 | idx2,:]
    lab27 = labels[idx1 | idx2]
    lab27[lab27==2] = 0
    lab27[lab27==7] = 1

    w = batch_grad_descent(img27, lab27, 0)
    plt.imshow(w.reshape(28, -1, order='F'), cmap='bone')
    print('Look look it is a pretty two i think')

if __name__ == '__main__':
    main()