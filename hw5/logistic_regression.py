import numpy as np
import data as dt
import helper_methods as hp

# np.random.seed(1)

class LogisticRegression:
    """
    Logistic Regression Model
    """
    # LR_RATES = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    # VAR_RATES = [0.1, 1, 10, 100, 1000, 10000]
    LR_RATES = [0.01, 0.001, 0.0001]
    VAR_RATES = [1, 10, 1000]

    def __init__(self):
        self._W = np.zeros(219, )       # Weight vector D x 1
        self._b = 0                     # the bias

        self._lr = None                 # learning rate hyperparameter
        self._var = None                # variance hyperparameter

        self._cross_val_f = None
        self._cross_val_p = None
        self._cross_val_r = None

        # find best hyperparameter
        self._best_learning_rate(LogisticRegression.LR_RATES, LogisticRegression.VAR_RATES)


    def train(self, train_data=None, epochs=20):
        # insatiate required data
        if train_data is None:
            train_data = dt.load_data(dt.TRAIN)
        hp._reset_parameters(self, self._W.shape[0])

        initial_lr = self._lr
        # train classifier over epochs
        for t in range(epochs):
            self._lr = hp.lr_svm(initial_lr, t)
            np.random.shuffle(train_data)

            for train_example in train_data:
                y = train_example[0]
                x = train_example[1:]

                dot_product = np.dot(self._W.T, x)
                if 1 + np.exp(-y * dot_product) > 0:
                    self._update(train_example, has_loss=True)
                else:
                    self._update(train_example, has_loss=False)
        self._lr = initial_lr

    # 1 / (1 + exp(-w*x))
    # returns false if prediction is right, and true if prediction is wrong
    def predict(self, test_data):
        y = test_data[0]
        x = test_data[1:]

        dot_product = np.dot(self._W.T, x)
        exponent = dot_product + self._b
        probabilty = 1 / (1 + np.exp(-exponent)) # given x and w what is the probability y == 1

        prediction = 1 if probabilty > 0.5 else -1

        return prediction != y


    # w =  (1-lr)w + lr C y x
    # b =  (1-lr)b + lr C y
    def _update(self, test_data, has_loss):
        y = test_data[0]
        x = test_data[1:]

        self._W = (1 - 2 * self._lr / self._var) * self._W
        self._b = (1 - 2 * self._lr / self._var) * self._b

        if has_loss:
            dot_product = np.dot(self._W.T, x)
            self._W += self._lr * y * x * np.exp(-y * dot_product) / (np.log(2) * (1 + np.exp(-y * dot_product)))
            self._b += self._lr * y * np.exp(-y * dot_product) / (np.log(2) * (1 + np.exp(-y * dot_product)))


    # find the best hyperparameter
    def _best_learning_rate(self, lr_rates, var_rates):
        accuracies = {}

        for lr, var in [(lr, var) for lr in lr_rates for var in var_rates]:
            self._lr, self._var = lr, var
            accuracies[(lr, var)] = hp._cross_validation_accuracy(self)

        self._lr, self._var = max(accuracies, key=accuracies.get)

        self._cross_val_f = accuracies[(self._lr, self._var)][0]
        self._cross_val_p = accuracies[(self._lr, self._var)][1]
        self._cross_val_r = accuracies[(self._lr, self._var)][2]

    def __str__(self):
        return 'Logistic Regression'
