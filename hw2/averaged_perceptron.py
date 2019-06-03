import numpy as np
import data as dt

# np.random.seed(1)

class AveragedPerceptron:

    def __init__(self):
        # self._X = None  # training examples N x 1+M (first column is label)
        self._W = None  # Weight vector M x 1
        self._b = None  # the bias
        self._A = None
        self._ba = None
        self._lr = None   # best learning rate hyperparameter
        self._reset_parameters()

        self._cross_val_acc = 0.0
        self._total_updates = 0
        self._dev_acc = 0.0
        self._epoch_acc = []

    def train(self, learning_rates):
        # find best hyperparameter
        self._lr = self._best_learning_rate(learning_rates)

        # train using best hyperparamter to find best A
        self._A, self._ba = self._best_classifier()

    # y(wTx + b) < 0
    def predict(self, test_data, use_avg_classifier=False):
        assert type(test_data) is np.ndarray, " test_data must be a np.ndarray/vector"

        y = test_data[0]
        x = test_data[1:]

        if use_avg_classifier:
            dot_product = np.dot(self._A.T, x)
            prediction = y * (dot_product + self._ba)
        else:
            dot_product = np.dot(self._W.T, x)
            prediction = y*(dot_product + self._b)

        return prediction < 0

    # w =  w + ryx,
    # b =  b + ry
    def _update(self, test_data, lr):
        y = test_data[0]
        x = test_data[1:]

        self._W = self._W + lr * y * x
        self._b = self._b + lr * y

    # find and return the best hyperparameter
    def _best_learning_rate(self, learning_rates):
        accuracies = {}

        for lr in learning_rates:
            accuracies[lr] = self._cross_validation_accuracy(lr)

        best_lr = max(accuracies, key=accuracies.get)
        self._cross_val_acc = accuracies[best_lr]

        return best_lr

    # Compute cross validation accuracy for specific learning rate: lr
    def _cross_validation_accuracy(self, lr):
        datas = []
        avg_accuracies = []

        # choosing i to be the test_fold and the rest to be the training folds
        for i in range(len(dt.FOLDS)):
            self._reset_parameters()
            test_data, train_data = None, None

            for index, filename in enumerate(dt.FOLDS):
                if i == index:
                    test_data = dt.load_data(filename)
                    continue
                datas.append(dt.load_data(filename))

            train_data = np.concatenate(datas)

            # starting training over 10 epochs
            for t in range(10):
                # np.random.seed(1)
                np.random.shuffle(train_data)

                for train_example in train_data:
                    if self.predict(train_example):
                        self._update(train_example, lr)

            wrong_predictions = 0
            for test_example in test_data:
                if self.predict(test_example):
                    wrong_predictions += 1

            accuracy = 1.0 - (wrong_predictions / len(test_data))
            avg_accuracies.append(accuracy)

        return np.mean(avg_accuracies)

    # best classifier over 20 epochs
    def _best_classifier(self):
        train_data = dt.load_data(dt.TRAIN)
        dev_data = dt.load_data(dt.DEV)

        self._reset_parameters()
        self._total_updates = 0

        best_classifier = np.zeros(19, )
        best_bias = 0.0
        best_accuracy = 0.0

        # train classifier over 20 epochs
        for t in range(20):
            # np.random.seed(1)
            np.random.shuffle(train_data)

            for train_example in train_data:
                if self.predict(train_example):
                    self._update(train_example, self._lr)
                    self._total_updates += 1
                self._A += self._W
                self._ba += self._b

            wrong_predictions = 0
            for dev_example in dev_data:
                if self.predict(dev_example, use_avg_classifier=True):
                    wrong_predictions += 1

            accuracy = 1.0 - (wrong_predictions / len(dev_data))

            self._epoch_acc.append((t, accuracy))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = self._A
                best_bias = self._ba

        self._dev_acc = best_accuracy

        return best_classifier, best_bias

    # reset weight and bias to something between -0.01 and 0.01
    def _reset_parameters(self):
        # np.random.seed(1)
        init_weight = np.random.uniform(low=-0.01, high=0.01)
        self._W = np.full((19, ), init_weight)
        self._b = init_weight
        self._A = np.full((19,), init_weight)
        self._ba = init_weight

    def report(self):
        test_data = dt.load_data(dt.TEST)

        wrong_predictions = 0
        for test_example in test_data:
            if self.predict(test_example, use_avg_classifier=True):
                wrong_predictions += 1

        accuracy = 1.0 - (wrong_predictions / len(test_data))

        print('*********************************')
        print('****** Averaged Perceptron ******')
        print('*********************************')
        print('(a) Best Learning Rate: {}'.format(self._lr))
        print('(b) Best Cross-Val Accuracy: {}'.format(self._cross_val_acc))
        print('(c) Total Nr. of updates: {}'.format(self._total_updates))
        print('(d) Development set accuracy: {}'.format(self._dev_acc))
        print('(e) Test set accuracy: {}'.format(accuracy))
        print('*********************************\n')
