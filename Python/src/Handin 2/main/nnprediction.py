import numpy as np
import matplotlib.pyplot as plt
import tensorflow_learner.TensorflowLearner as tfl
import validation.ModelEvaluator as eval
import tensorflow_learner.Models as models
import scikit_learner as skl

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

def runNNExperiment():
    plt.interactive(True)
    mnist_train_img, mnist_train_lab = load_data_au('data/auTrain.npz')
    mnist_test_img, mnist_test_lab = load_data_au('data/auTest.npz')

    model = models.SimpleNeuralNet()

    recognizer = tfl.ImageRecognizer(model)

    recognizer.train(mnist_train_img, mnist_train_lab)
    predictions = recognizer.predict(mnist_test_img)

    evaluator = eval.ModelEvaluator()
    report = evaluator.create_report(predictions.astype(np.int32), mnist_test_lab.astype(np.int32))
    error_rate = evaluator.get_error_rate(predictions.astype(np.int32), au_test_lab.astype(np.int32))

    print("Error Rate: ")
    print(error_rate)

    print("Detailed classification report:")
    print(report)

    print("Training time: {} seconds".format(recognizer.trainingtime))


def runSVNExperiments():
    plt.interactive(True)
    au_train_img, au_train_lab = load_data_au('data/auTrain.npz')
    au_test_img, au_test_lab = load_data_au('data/auTest.npz')
    recognizer = skl.CrossvalidatedSVMLearner()

    recognizer.train(au_train_img, au_train_lab)
    predictions = recognizer.predict(au_test_img)

    evaluator = eval.ModelEvaluator()
    report = evaluator.create_report(predictions.astype(np.int32), au_test_lab.astype(np.int32))
    error_rate = evaluator.get_error_rate(predictions.astype(np.int32), au_test_lab.astype(np.int32))

    print("Error Rate: ")
    print(error_rate)

    print("Detailed classification report:")
    print(report)

    print("Training time: {} seconds".format(recognizer.trainingtime))

if __name__ == "__main__":
    runNNExperiment()

    # TODO:Refactor to include dropout rate
    # TODO:Log and save acuracy during training both insample and out of sample (after eeach epoch)
    # TODO:Implement 3 layer neural network
    # TODO:Implement an svm see assignment
    # TODO:Save svm
    # TODO:Tune model even more
    # TODO:Create prediction function to handin
    # TODO: Write report
