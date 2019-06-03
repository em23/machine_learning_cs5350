import numpy as np
import data as dt
import helper_methods as hp

class NaiveBayes:
    """
    Naive Bayes Model
    """
    SMOOTHING_RATES = [2, 1.5, 1.0, 0.5]

    # SMOOTHING_RATES = [2]

    def __init__(self):
        self._W_poz = np.zeros(74482,)       # Weight vector D x 1
        self._W_neg = np.zeros(74482,)       # Weight vector D x 1
        self._prior_poz = 0
        self._prior_neg = 0

        # self._s = None                    # smoothing hyperparameter
        self._s = 1.0
        self._verbose = True

        # find best hyperparameter
        # self._best_learning_rate(NaiveBayes.SMOOTHING_RATES)

        self._W_poz = np.zeros(74482, dtype=int)  # Weight vector D x 1
        self._W_neg = np.zeros(74482, dtype=int)  # Weight vector D x 1


    def train(self, train_data=None, epochs=20):
        # insatiate required data
        if train_data is None:
            train_data = dt.load_data(dt.TRAIN)

        for train_example in train_data:
            self._update(train_example)

        self._prior_poz = self._prior_poz / (self._prior_neg + self._prior_poz)
        self._prior_neg = 1.0 - self._prior_poz
        # print([x for x in self._W_poz])
        # print([x for x in self._W_neg])



    def predict(self, test_data, on_eval= False):
        y = test_data[0]
        x = test_data[1]

        prob_x_poz = self._compute_prob_x_given_y(x, self._W_poz, self._prior_poz)
        prob_x_neg = self._compute_prob_x_given_y(x, self._W_neg, self._prior_neg)

        pozitives, negatives = 0.0, 0.0
        for column, value in prob_x_poz.items():
            pozitives += value
        for column, value in prob_x_neg.items():
            negatives += value

        prediction = 1 if pozitives > negatives else -1

        if on_eval:
            return True if prediction != 1 else False
        return prediction != y


    def _compute_prob_x_given_y(self, x, count_x_y, count_y):
        S_i = 1 # the number of all possible values that xi can take in the data.

        x_neg = {}
        for column, value in x.items():
            x_neg[column] = count_x_y[column]

        count_x = {}
        for column, value in x.items():
            count_x[column] = x_neg[column] / count_y

        nominator_neg = {}
        for column, value in x.items():
            nominator_neg[column] = count_x[column] + self._s

        denominator_neg = count_y + S_i * self._s

        prob_x_neg = {}
        for column, value in x.items():
            prob_x_neg[column] = nominator_neg[column] / denominator_neg

        return prob_x_neg

    def _update(self, test_data):
        y = test_data[0]
        x = test_data[1]

        if y > 0:
            # self._W_poz += x
            for column, value in x.items():
                self._W_poz[column] += 1
            self._prior_poz += 1
        else:
            # self._W_neg += x
            for column, value in x.items():
                self._W_neg[column] += 1
            self._prior_neg += 1


    # find the best hyperparameter
    def _best_learning_rate(self, somoothing_rates):
        accuracies = {}

        for smoothing in somoothing_rates:
            self._s = smoothing
            accuracies[smoothing] = hp._cross_validation_accuracy(self)

        self._s = max(accuracies, key=accuracies.get)


    def __str__(self):
        return 'Naive Bayes'
