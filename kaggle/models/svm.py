import numpy as np
import data as dt
import helper_methods as hp

class SVM:
    """
    Support Vector Machine Model
    # """
    LR_RATES = [1, 0.1, 0.01, 0.001, 0.0001]
    C_RATES = [10, 1, 0.1, 0.01, 0.001, 0.0001]

    # LR_RATES = [0.000001]
    # C_RATES = [1]

    def __init__(self, D=74482, verbose=False):
        self._W = np.zeros(D, )       # Weight vector D x 1
        self._b = 0                     # the bias

        self._lr = None                 # learning rate hyperparameter
        self._c = None                  # regularization hyperparameter

        self._verbose = verbose

        self._best_learning_rate(SVM.LR_RATES, SVM.C_RATES)

    def train(self, train_data=None, epochs=20):
        # insatiate required data
        if train_data is None:
            train_data = dt.load_data(dt.TRAIN)

        initial_lr = self._lr
        # train classifier over epochs
        for t in range(epochs):
            self._lr = hp.lr_svm(initial_lr, t)
            np.random.shuffle(train_data)

            if self._verbose:
                wrong_predictions = 0

            for train_example in train_data:
                if self.predict(train_example):
                    self._update(train_example, has_loss=True)
                    if self._verbose:
                        wrong_predictions += 1
                else:
                    self._update(train_example, has_loss=False)

            if self._verbose:
                accuracy = 1.0 - (wrong_predictions / len(train_data))
                print('Epoch: {} | Accuracy: {}%'.format(t, accuracy))

        self._lr = initial_lr

    # y(wTx + b) <= 1
    def predict(self, test_data):
        y = test_data[0]
        x = test_data[1]

        # dot_product = np.dot(self._W.T, x)
        dot_product = 0
        for column, value in x.items():
            dot_product += self._W[column] * value
        prediction = y * (dot_product + self._b)

        return prediction <= 0


    # w =  (1-lr)w + lr C y x
    # b =  (1-lr)b + lr C y
    def _update(self, test_data, has_loss):
        y = test_data[0]
        x = test_data[1]

        self._W = (1 - self._lr) * self._W
        self._b = (1 - self._lr) * self._b

        if has_loss:
            # self._W += self._lr * self._c * y * x
            for column, value in x.items():
                self._W[column] += self._lr * self._c * y * value
            self._b += self._lr * self._c * y


    # find the best hyperparameter
    def _best_learning_rate(self, lr_rates, c_rates):
        accuracies = {}

        for lr, c in [(lr, c) for lr in lr_rates for c in c_rates]:
            self._lr, self._c = lr, c
            accuracies[(lr, c)] = hp._cross_validation_accuracy(self, verbose=self._verbose)
            if self._verbose:
                print('LR: {}, C: {} | Cross-Validation Accuracy: {}%'.format(lr, c, accuracies[(lr, c)]))

        self._lr, self._c = max(accuracies, key=accuracies.get)

    def __str__(self):
        return 'SVM'
