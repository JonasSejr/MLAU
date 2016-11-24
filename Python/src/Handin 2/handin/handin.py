import numpy as np
import tensorflow as tf


class Model( object ):
    def constructTrainModel( self, d ):
        raise NotImplementedError( "Should have implemented this" )


    def constructPredictModel(self, d):
        raise NotImplementedError("Should have implemented this")


class DeepConvolutionalNet(Model):
    def __init__(self, hidden_layer_breadth, drop_out_keep_prop, conv_channels):
        self.hidden_layer_breadth = hidden_layer_breadth
        self.drop_out_keep_prop = drop_out_keep_prop
        self.conv_channels = conv_channels

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

    def construct(self, d, keep_prob):
        X = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.int32, shape=[None])

        W_conv1 = self.weight_variable([5, 5, 1, self.conv_channels])
        b_conv1 = self.bias_variable([self.conv_channels])

        x_image = tf.reshape(X, [-1,28,28,1])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)# Reduction to image size 14 * 14

        W_conv2 = self.weight_variable([5, 5, self.conv_channels, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)# Reduction to image size 7 * 7
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])


        beta = tf.Variable(tf.constant(0.0, shape=[7 * 7 * 64,]))
        gamma = tf.Variable(tf.constant(1.0, shape=[7 * 7 * 64,]))
        mean, variance = tf.nn.moments(h_pool2_flat, [0])
        batch_normalized = tf.nn.batch_normalization(h_pool2_flat, mean, variance, beta, gamma, 1e-3)#Normalization used for both training and prediction

        W_fc1 = self.weight_variable([7 * 7 * 64, self.hidden_layer_breadth])#was 1024
        b_fc1 = self.bias_variable([self.hidden_layer_breadth])
        x = tf.matmul(batch_normalized, W_fc1) + b_fc1
        h_fc1 = tf.nn.relu(x)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = self.weight_variable([self.hidden_layer_breadth, 10])
        b_fc2 = self.bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        y = tf.cast(tf.argmax(y_conv, 1), tf.int32)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y_)
        mean_cross_entropy = tf.reduce_mean(cross_entropy)
        regularized_loss = mean_cross_entropy #No regularization
        return X, y_, y, regularized_loss


    def constructTrainModel(self, d):
        return self.construct(d, self.drop_out_keep_prop)


    def constructPredictModel(self, d):
        return self.construct(d, 1)


def predict(images):
    model = DeepConvolutionalNet(hidden_layer_breadth=1000, drop_out_keep_prop=0.5, conv_channels=40)
    filename = 'bestmodel.tf'
    with tf.Graph().as_default():
        n, d = images.shape
        X, Y_, Y, regularized_loss = model.constructPredictModel(d)
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, filename)
            return session.run(Y, feed_dict={X: images})


def load_data_au(filename):
    train_file = np.load(filename)
    images = train_file['digits']  # image
    labels = train_file['labels']
    return images, labels

if __name__ == "__main__":
    #test predict function
    au_test_img, au_test_lab = load_data_au('../data/auTest.npz')
    prediction = predict(au_test_img)
    print(np.mean(prediction == au_test_lab))