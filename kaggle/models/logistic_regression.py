import numpy as np
import data as dt
import helper_methods as hp


class LogisticRegression:
    """
    Logistic Regression Model
    """
    LR_RATES = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    VAR_RATES = [0.1, 1, 10, 100, 1000, 10000]

    # LR_RATES = [0.00001]
    # VAR_RATES = [1, 10]

    def __init__(self, verbose=False):
        self._W = np.zeros(74482, )     # Weight vector D x 1
        self._b = 0                     # the bias

        self._lr = None                 # learning rate hyperparameter
        self._var = None                # variance hyperparameter

        self._verbose = verbose

        # find best hyperparameter
        self._best_learning_rate(LogisticRegression.LR_RATES, LogisticRegression.VAR_RATES)


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
                y = train_example[0]
                x = train_example[1]

                if self._verbose:
                    if self.predict(train_example):
                        wrong_predictions += 1

                # dot_product = np.dot(self._W.T, x)
                # dot_product = self.custom_dot_product(x)
                # if 1 + np.exp(-y * dot_product) > 0:
                if self.predict(train_example):
                    self._update(train_example, has_loss=True)
                else:
                    self._update(train_example, has_loss=False)

            if self._verbose:
                accuracy = 1.0 - (wrong_predictions / len(train_data))
                print('Epoch: {} | Accuracy: {}%'.format(t, accuracy))

        self._lr = initial_lr

    # 1 / (1 + exp(-w*x))
    # returns false if prediction is right, and true if prediction is wrong
    def predict(self, test_data):
        y = test_data[0]
        x = test_data[1]

        # dot_product = np.dot(self._W.T, x)
        dot_product = self.custom_dot_product(x)
        exponent = dot_product + self._b
        # probabilty = 1 / (1 + np.exp(-exponent)) # given x and w what is the probability y == 1
        # prediction = 1 if probabilty > 0.5 else -1

        #sgn(exponent)
        prediction = 1 if exponent > 0 else -1

        return prediction != y


    # w =  (1-lr)w + lr C y x
    # b =  (1-lr)b + lr C y
    def _update(self, test_data, has_loss):
        y = test_data[0]
        x = test_data[1]

        self._W = (1 - 2 * self._lr / self._var) * self._W
        self._b = (1 - 2 * self._lr / self._var) * self._b

        if has_loss:
            # dot_product = np.dot(self._W.T, x)
            dot_product = self.custom_dot_product(x)
            for column, value in x.items():
                self._W[column] += self._lr * y * value * 1 /  (1 + np.exp(y * dot_product))
            self._b += self._lr * y * 1 / (1 + np.exp(y * dot_product))
            # self._W += self._lr * y * x * np.exp(-y * dot_product) / (np.log(2) * (1 + np.exp(-y * dot_product)))
            # self._b += self._lr * y * np.exp(-y * dot_product) / (np.log(2) * (1 + np.exp(-y * dot_product)))


    # find the best hyperparameter
    def _best_learning_rate(self, lr_rates, var_rates):
        accuracies = {}

        for lr, var in [(lr, var) for lr in lr_rates for var in var_rates]:
            self._lr, self._var = lr, var
            accuracies[(lr, var)] = hp._cross_validation_accuracy(self)
            if self._verbose:
                print('LR: {}, TRADEOFF: {} | Cross-Validation Accuracy: {}%'.format(lr, var, accuracies[(lr, var)]))

        self._lr, self._var = max(accuracies, key=accuracies.get)


    def custom_dot_product(self, x):
        dot_product = 0
        for column, value in x.items():
            dot_product += self._W[column] * value
        return dot_product


    def __str__(self):
        return 'Logistic Regression'
