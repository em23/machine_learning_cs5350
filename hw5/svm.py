import numpy as np
import data as dt
import helper_methods as hp

class SVM:
    """
    Support Vector Machine Model
    # """
    LR_RATES = [1, 0.1, 0.01, 0.001, 0.0001]
    C_RATES = [10, 1, 0.1, 0.01, 0.001, 0.0001]


    def __init__(self, D=219):
        self._W = np.zeros(D, )       # Weight vector D x 1
        self._b = 0                     # the bias

        self._lr = None                 # learning rate hyperparameter
        self._c = None                  # regularization hyperparameter

        self._cross_val_f = None
        self._cross_val_p = None
        self._cross_val_r = None

        # find best hyperparameter
        if D != 219:
            self._lr = 0.0001
            self._c = 10
        else:
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

            for train_example in train_data:
                if self.predict(train_example):
                    self._update(train_example, has_loss=True)
                else:
                    self._update(train_example, has_loss=False)
        self._lr = initial_lr

    # y(wTx + b) <= 1
    def predict(self, test_data):
        y = test_data[0]
        x = test_data[1:]

        dot_product = np.dot(self._W.T, x)
        prediction = y * (dot_product + self._b)

        return prediction <= 0


    # w =  (1-lr)w + lr C y x
    # b =  (1-lr)b + lr C y
    def _update(self, test_data, has_loss):
        y = test_data[0]
        x = test_data[1:]

        self._W = (1 - self._lr) * self._W
        self._b = (1 - self._lr) * self._b

        if has_loss:
            self._W += self._lr * self._c * y * x
            self._b += self._lr * self._c * y


    # find the best hyperparameter
    def _best_learning_rate(self, lr_rates, c_rates):
        accuracies = {}

        for lr, c in [(lr, c) for lr in lr_rates for c in c_rates]:
            self._lr, self._c = lr, c
            accuracies[(lr, c)] = hp._cross_validation_accuracy(self)

        self._lr, self._c = max(accuracies, key=accuracies.get)

        self._cross_val_f = accuracies[(self._lr, self._c)][0]
        self._cross_val_p = accuracies[(self._lr, self._c)][1]
        self._cross_val_r = accuracies[(self._lr, self._c)][2]

    def __str__(self):
        return 'SVM'
