import tensorflow as tf

class Model( object ):
    def construct( self, d ):
        raise NotImplementedError( "Should have implemented this" )

class Softmax(Model):
    def construct(self, d):
        reg_rate = 1e-4
        # parameters for the learning algorithm
        X = tf.placeholder(tf.float32, shape=[None, d])
        Y_ = tf.placeholder(tf.int32, shape=[None])
        # Parameters for the model to learn. These are the ones to  optimize based on the errorterm
        W = tf.Variable(tf.zeros([d, 10]))
        b = tf.Variable(tf.zeros([10]))
        # Express the formula
        logits = tf.matmul(X, W) + b  # We skip the calculation of the softmax function
        Y = tf.cast(tf.argmax(logits, 1), Y_.dtype)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, Y_)
        loss = tf.reduce_mean(cross_entropy)
        # L_2 weight decay
        reg = reg_rate * tf.reduce_sum(W ** 2)
        # Minimization target is the sum of cross-entropy loss and regularization
        regularized_loss = loss + reg
        return X, Y_, Y, regularized_loss

class DeepConvolutionalNet(Model):


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def construct(self, d):
        reg_rate = 0

        X = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.int32, shape=[None])

        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])

        x_image = tf.reshape(X, [-1,28,28,1])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)# Reduction to image size 14 * 14

        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)# Reduction to image size 7 * 7

        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

        y = tf.cast(tf.argmax(y_conv, 1), tf.int32)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y_)
        mean_cross_entropy = tf.reduce_mean(cross_entropy)
        regularized_loss = mean_cross_entropy #No regularization
        return X, y_, y, regularized_loss


class SimpleNeuralNet(Model):

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def construct(self, d):
        reg_rate = 0
        #Add layer one bredth as a variable
        X = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.int32, shape=[None])

        W_1 = self.weight_variable([784, 784])
        b_1 = self.bias_variable([784])

        L1_out = tf.nn.relu(tf.matmul(X, W_1) + b_1)

        #keep_prob = tf.placeholder(tf.float32)
        #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_2 = self.weight_variable([784, 10])
        b_2 = self.bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(L1_out, W_2) + b_2)

        y = tf.cast(tf.argmax(y_conv, 1), tf.int32)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y_)
        mean_cross_entropy = tf.reduce_mean(cross_entropy)
        regularized_loss = mean_cross_entropy #No regularization
        return X, y_, y, regularized_loss

class DeepNeuralNet(Model):

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def add_layer(self, X, dimension_in, dimension_out):
        W_1 = self.weight_variable([dimension_in, dimension_out])
        b_1 = self.bias_variable([dimension_out])
        L1_out = tf.nn.relu(tf.matmul(X, W_1) + b_1)
        return L1_out

    def construct(self, d):
        reg_rate = 0
        #Add layer one bredth as a variable
        X = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.int32, shape=[None])

        L1_out = self.add_layer(X)

        #keep_prob = tf.placeholder(tf.float32)
        #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_2 = self.weight_variable([784, 10])
        b_2 = self.bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(L1_out, W_2) + b_2)

        y = tf.cast(tf.argmax(y_conv, 1), tf.int32)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y_)
        mean_cross_entropy = tf.reduce_mean(cross_entropy)
        regularized_loss = mean_cross_entropy #No regularization
        return X, y_, y, regularized_loss

