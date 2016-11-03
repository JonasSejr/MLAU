import numpy as np
import matplotlib.pyplot as plt
import tensorflow_learner.TensorflowLearner as tfl
import validation.ModelEvaluator as eval
import tensorflow_learner.Models as models
from sklearn.model_selection import train_test_split
import scikit_learner as skl
import scipy.ndimage
from scipy.ndimage.interpolation import shift

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

def randomizeImage(image):
    as_image = image.reshape(28,-1)
    rotation = np.random.uniform(-15, 15)
    shift_x = np.random.uniform(-5, 5)
    shift_y = np.random.uniform(-5, 5)
    rotated_images = scipy.ndimage.rotate(as_image, rotation, reshape=False)
    randomized_image  = shift(rotated_images, [shift_x, shift_y])
    randomized_image_flattend = randomized_image.flatten()
    return randomized_image_flattend

def expandTrainingSet(images, labs):
    img_1 = np.apply_along_axis(randomizeImage, 1, images)
    img_2 = np.apply_along_axis(randomizeImage, 1, images)
    img_3 = np.apply_along_axis(randomizeImage, 1, images)
    img_4 = np.apply_along_axis(randomizeImage, 1, images)

    expanded_images = np.concatenate((images, img_1, img_2, img_3, img_4))
    expanded_labs = np.concatenate((labs, labs, labs, labs, labs))
    return expanded_images, expanded_labs



def runNNExperiment():
    plt.interactive(True)
    au_train_orig_img, au_train_orig_lab = load_data_au('data/auTrain.npz')
    au_train_img, au_valid_img, au_train_lab, au_valid_lab = train_test_split(au_train_orig_img, au_train_orig_lab, test_size=0.2, random_state=0)


    au_test_img, au_test_lab = load_data_au('data/auTest.npz')
    #au_train_expanded_img, au_train_expanded_lab = expandTrainingSet(au_train_img, au_train_lab)

    #model = models.DeepConvolutionalNet(hidden_layer_breadth=1000, drop_out_keep_prop=0.5, conv_channels=40)
    model = models.SimpleNeuralNet(hidden_layer_breadth=1084, drop_out_keep_prop=1, reg_rate=0)

    recognizer = tfl.ImageRecognizer(model)
    recognizer.train(au_train_img, au_train_lab)

    evaluator = eval.ModelEvaluator()

    predictions = recognizer.predict(au_valid_img)
    report = evaluator.create_report(predictions.astype(np.int32), au_valid_lab.astype(np.int32))
    error_rate = evaluator.get_error_rate(predictions.astype(np.int32), au_valid_lab.astype(np.int32))

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
