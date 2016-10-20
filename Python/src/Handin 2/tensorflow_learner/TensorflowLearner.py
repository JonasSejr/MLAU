import tensorflow as tf
import numpy as np
import time

class ImageRecognizer:

    def __init__(self, model):
        self.model = model
        self.trainingtime = 0

    def permute_data_all(self, images, labels):
        assert images.flags.c_contiguous
        perm = np.random.permutation(len(labels))
        return images[perm], labels[perm]

    def create_tf_optimizer(self, regularized_loss):
        learning_rate = tf.placeholder(tf.float32)
        opt = tf.train.AdamOptimizer(learning_rate)
        train_step = opt.minimize(regularized_loss)
        return learning_rate, train_step

    def optimize_with_minibatch(self, X, y_, y, learning_rate, session, tf_optimizer,
                                        train_images, train_labels):
        epochs = 100
        batch_size = 50
        correct_prediction = tf.equal(y, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        session.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            input_value, labels_value = self.permute_data_all(train_images, train_labels)

            for i in range(0, len(train_labels), batch_size):
                j = min(len(train_labels), i + batch_size)
                endof_next_batch = min(len(train_labels), j + batch_size)
                if i % (4 * batch_size) == 0 & j != len(train_labels):
                    train_accuracy = accuracy.eval(feed_dict={
                        X: input_value[j:endof_next_batch],
                        y_: labels_value[j:endof_next_batch]})
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                    session.run(
                        tf_optimizer,
                        feed_dict={X: input_value[i:j],
                                   y_: labels_value[i:j],
                                   learning_rate: 1e-4})

    def train(self, train_images, train_labels):
        start_time = time.time()

        with tf.Graph().as_default():
            d = train_images.shape[1]
            X, y_, y, regularized_loss= self.model.construct(d)
            saver = tf.train.Saver()
            learning_rate, tf_optimizer = self.create_tf_optimizer(regularized_loss)

            with tf.Session() as session:
                self.optimize_with_minibatch(X, y_, y, learning_rate, session, tf_optimizer,
                                        train_images, train_labels)
                saver.save(session, "mymodel.tf")
        self.trainingtime = time.time() - start_time


    def predict(self, data_value):
        filename = 'mymodel.tf'
        with tf.Graph().as_default():
            n, d = data_value.shape
            X, Y_, Y, regularized_loss = self.model.construct(d)
            saver = tf.train.Saver()
            with tf.Session() as session:
                saver.restore(session, filename)
                return session.run(Y, feed_dict={X: data_value})
