from builtins import print
from sklearn import svm
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


class CrossvalidatedSVMLearner:

    def train(self, train_images, train_labels):
        start_time = time.time()
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.0001], 'C': [1, 10, 100, 1000]},
                            #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}#,
                            #{'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'degree': [3]}
            ]

        # scores = ['precision', 'recall']
        score = 'precision'

        print("# Tuning hyper-parameters for %s" % score)
        print()

        self.clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        self.clf.fit(train_images, train_labels)

        print("Best parameters set found on development set:")
        print()
        print(self.clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = self.clf.cv_results_['mean_test_score']
        stds = self.clf.cv_results_['std_test_score']
        params = self.clf.cv_results_['params']
        for mean, std, params in zip(means, stds, params):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        self.trainingtime = time.time() - start_time

    def predict(self, data_value):
        return self.clf.predict(data_value)