import numpy as np
import data as dt
import helper_methods as hp

class NaiveBayes:
    """
    Naive Bayes Model
    """
    SMOOTHING_RATES = [2, 1.5, 1.0, 0.5]

    def __init__(self):
        self._W_poz = np.zeros(219,)       # Weight vector D x 1
        self._W_neg = np.zeros(219,)       # Weight vector D x 1
        self._prior_poz = 0
        self._prior_neg = 0

        self._s = None                    # smoothing hyperparameter

        self._cross_val_f = None
        self._cross_val_p = None
        self._cross_val_r = None

        # find best hyperparameter
        self._best_learning_rate(NaiveBayes.SMOOTHING_RATES)


    def train(self, train_data=None, epochs=20):
        # insatiate required data
        if train_data is None:
            train_data = dt.load_data(dt.TRAIN)

        for train_example in train_data:
            self._update(train_example)



    def predict(self, test_data):
        y = test_data[0]
        x = test_data[1:]

        prob_x_poz = self._compute_prob_x_given_y(x, self._W_poz, self._prior_poz)
        prob_x_neg = self._compute_prob_x_given_y(x, self._W_neg, self._prior_neg)

        prediction = 1 if prob_x_poz.sum() > prob_x_neg.sum() else -1

        return prediction != y


    def _compute_prob_x_given_y(self, x, count_x_y, count_y):
        S_i = 2  # the number of all possible values that xi can take in the data.

        x_neg = count_x_y * x
        count_x_neg = x_neg / count_y
        nominator_neg = count_x_neg + self._s
        denominator_neg = count_y + S_i * self._s
        prob_x_neg = nominator_neg / denominator_neg

        return prob_x_neg


    def _update(self, test_data):
        y = test_data[0]
        x = test_data[1:]

        if y > 0:
            self._W_poz += x
            self._prior_poz += 1
        else:
            self._W_neg += x
            self._prior_neg += 1


    # find the best hyperparameter
    def _best_learning_rate(self, somoothing_rates):
        accuracies = {}

        for smoothing in somoothing_rates:
            self._s = smoothing
            accuracies[smoothing] = hp._cross_validation_accuracy(self)

        self._s = max(accuracies, key=accuracies.get)

        self._cross_val_f = accuracies[self._s][0]
        self._cross_val_p = accuracies[self._s][1]
        self._cross_val_r = accuracies[self._s][2]

    def __str__(self):
        return 'Naive Bayes'
