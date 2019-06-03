import data as dt
import numpy as np
import csv

# np.random.seed(1)

"""
TP - true positive
FP - false positive
FN - false negative
"""

# is true positive
is_tp = lambda prediction, y_true: not prediction and y_true > 0.0
# is false positive
is_fp = lambda prediction, y_true: prediction and y_true < 0.0
# is false negative
is_fn = lambda prediction, y_true: prediction and y_true > 0.0

# precision
p = lambda tp, fp: tp / (tp + fp) if tp + fp != 0 else 0.0
# recall
r = lambda tp, fn: tp / (tp + fn) if tp + fn != 0 else 0.0
# f-value
f = lambda p, r: 2 * p * r / (p + r) if p + r != 0 else 0.0

# svm learning rate
lr_svm = lambda gamma_0, t: gamma_0 / (1 + t)


# Compute cross validation accuracy for specific learning rate using specified classifier
# return f-socre, precision, recall
def _cross_validation_accuracy(model, verbose=False):
    avg_acc = []
    if hasattr(model, '_W'):
        D = model._W.shape[0]  # dimension
    else:
        D = 74482

    # choosing i to be the test_fold and the rest to be the training folds
    for i in range(len(dt.FOLDS)):
        _reset_parameters(model, D)
        # train_data = np.ndarray((0, D)) # +1 for the label
        test_data, train_data = [], []

        for index, filename in enumerate(dt.FOLDS):
            if i == index:
                test_data = dt.load_data(filename)
                continue
            # train_data = np.concatenate([train_data, dt.load_data(filename)], axis=0)
            train_data.extend(dt.load_data(filename))

        model.train(train_data=train_data, epochs=1)

        wrong_predictions = 0
        for test_example in test_data:
            prediction = model.predict(test_example) # true is wrong, false is correct
            if prediction:
                wrong_predictions += 1

        accuracy = 1.0 - (wrong_predictions / len(test_data))
        avg_acc.append(accuracy)

    _reset_parameters(model, D)
    return np.mean(avg_acc)


# reset weight and bias to something between low and high
def _reset_parameters(model, dimension, low=0.0, high=0.0):
    init_weight = np.random.uniform(low, high)
    if hasattr(model, '_W'):
        model._W = np.full((dimension,), init_weight)
        model._b = init_weight
    if hasattr(model, '_W_poz'):
        model._W_poz = np.full((dimension,), init_weight)
        model._W_neg = np.full((dimension,), init_weight)
        model._prior_poz = init_weight
        model._prior_neg = init_weight

# reporting results
def report(model, test_data=None):
    if test_data is None:
        test_data = dt.load_data(dt.TEST)

    if model._verbose:
        total = len(test_data)
        completed = 0
        print('######### Final Report ##########')
        print('Testing it on the test dataset')

    wrong_predictions = 0
    for test_example in test_data:
        prediction = model.predict(test_example)  # true is wrong, false is correct

        if prediction:
            wrong_predictions += 1

        if model._verbose:
            completed += 1
            if completed == int(0.75 * total):
                print("Processed 75% of dataset")
            elif completed == int(0.55 * total):
                print("Processed 50% of dataset")
            elif completed == int(0.25 * total):
                print("Processed 25% of dataset")


    accuracy = 1.0 - (wrong_predictions / len(test_data))

    print('*******************************')
    print('****** {} ******'.format(model))
    print('*******************************')
    print('Accuracy: {}'.format(accuracy))
    if hasattr(model, '_lr'):
        print('Best Learning Rate: {}'.format(model._lr))
    if hasattr(model, '_c'):
        print('Best Regularization: {}'.format(model._c))
    if hasattr(model, '_var'):
        print('Best Tradeoff: {}'.format(model._var))
    if hasattr(model, '_s'):
        print('Best Smoothing: {}'.format(model._s))
    print('*******************************\n')

    return accuracy

def evaluate(model):
    test_data = dt.load_data(dt.EVAL)
    test_data_id = dt.np_load_data(dt.EVAL_ID)

    with open('eval.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, quotechar=' ')
        spamwriter.writerow(["example_id", "label"])
        for i, test_example in enumerate(test_data):
            if hasattr(model, '_s'):
                prediction = model.predict(test_example, on_eval=True)
            else:
                prediction  = model.predict(test_example)

            if prediction:
                spamwriter.writerow([test_data_id[i], "0"])
            else:
                spamwriter.writerow([test_data_id[i],'1'])