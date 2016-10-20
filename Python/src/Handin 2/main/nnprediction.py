import numpy as np
import matplotlib.pyplot as plt
import tensorflow_learner.TensorflowLearner as tfl
import validation.ModelEvaluator as eval
import tensorflow_learner.Models as models

def load_data_au(filename):
    train_file = np.load(filename)
    images = train_file['digits']  # image
    labels = train_file['labels']
    return images, labels

def load_data_mnist(filename):
    train_file = np.load(filename)
    images = train_file['images']  # image
    labels = train_file['labels']
    return images, labels

if __name__ == "__main__":
    plt.interactive(True)
    mnist_train_img, mnist_train_lab = load_data_au('data/auTrain.npz')
    mnist_test_img, mnist_test_lab = load_data_au('data/auTest.npz')

    model = models.SimpleNeuralNet()

    recognizer = tfl.ImageRecognizer(model)

    recognizer.train(mnist_train_img, mnist_train_lab)
    predictions = recognizer.predict(mnist_test_img)

    evaluator = eval.ModelEvaluator()
    accuracy = evaluator.evaluate_logistic(predictions, mnist_test_lab.astype(np.int32))

    print("Errors on test set: {:.2%}".format(accuracy))

    print("Training time: {:.2} seconds".format(recognizer.trainingtime))


    # TODO:Refactor to include dropout rate
    # TODO:Log and save acuracy during training both insample and out of sample (after eeach epoch)
    # TODO:Implement 3 layer neural network
    # TODO:Implement an svm see assignment
    # TODO:Save svm
    # TODO:Tune model even more
    # TODO:Create prediction function to handin
    # TODO: Write report
