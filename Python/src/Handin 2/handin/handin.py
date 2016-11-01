import numpy as np
import tensorflow as tf
import tensorflow_learner.Models as models

class SimpleNeuralNet():

    def __init__(self, reg_rate, hidden_layer_breadth, drop_out_keep_prop):
        self.reg_rate = reg_rate
        self.hidden_layer_breadth = hidden_layer_breadth
        self.drop_out_keep_prop = drop_out_keep_prop

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def construct(self, d, keep_prob):
        #Add layer one breadth as a variable
        X = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.int32, shape=[None])

        W_1 = self.weight_variable([784, self.hidden_layer_breadth])
        b_1 = self.bias_variable([self.hidden_layer_breadth])

        L1_out = tf.nn.relu(tf.matmul(X, W_1) + b_1)

        L1_out_drop = tf.nn.dropout(L1_out, keep_prob)

        W_2 = self.weight_variable([self.hidden_layer_breadth, 10])
        b_2 = self.bias_variable([10])

        L2_out = tf.nn.softmax(tf.matmul(L1_out_drop, W_2) + b_2)

        y = tf.cast(tf.argmax(L2_out, 1), tf.int32)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(L2_out, y_)
        mean_cross_entropy = tf.reduce_mean(cross_entropy)
        # L_2 weight decay
        reg = self.reg_rate * (tf.reduce_sum(W_1 ** 2) + tf.reduce_sum(W_2 ** 2))
        regularized_loss = mean_cross_entropy + reg
        return X, y_, y, regularized_loss

    def constructTrainModel(self, d):
        return self.construct(d, self.drop_out_keep_prop)


    def constructPredictModel(self, d):
        return self.construct(d, 1)


def predict(images):
    model = models.DeepConvolutionalNet(hidden_layer_breadth=1000, drop_out_keep_prop=0.5, conv_channels=40)
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