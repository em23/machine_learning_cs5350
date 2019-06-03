import data as dt
import numpy as np

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
def _cross_validation_accuracy(model):
    D = 219
    if hasattr(model, '_W'):
        D = model._W.shape[0]  # dimension

    avg_p, avg_r, avg_f = [], [], []

    # choosing i to be the test_fold and the rest to be the training folds
    for i in range(len(dt.FOLDS)):
        _reset_parameters(model, D)
        train_data = np.ndarray((0, 220))
        test_data = None
        tp = fp = fn = 0

        for index, filename in enumerate(dt.FOLDS):
            if i == index:
                test_data = dt.load_data(filename)
                continue
            train_data = np.concatenate([train_data, dt.load_data(filename)], axis=0)

        model.train(train_data=train_data, epochs=10)

        for test_example in test_data:
            prediction =  model.predict(test_example) # true is wrong, false is correct
            y_true = test_example[0]

            if is_tp(prediction, y_true):
                tp += 1
            elif is_fp(prediction, y_true):
                fp += 1
            elif is_fn(prediction, y_true):
                fn += 1

        avg_p.append(p(tp, fp))
        avg_r.append(r(tp, fn))
        avg_f.append(f(avg_p[-1], avg_r[-1]))

    _reset_parameters(model, D)
    return np.mean(avg_f), np.mean(avg_p), np.mean(avg_r)


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

    wrong_predictions = tp = fp = fn = 0
    for test_example in test_data:
        prediction = model.predict(test_example)  # true is wrong, false is correct
        y_true = test_example[0]

        if prediction:
            wrong_predictions += 1

        if is_tp(prediction, y_true):
            tp += 1
        elif is_fp(prediction, y_true):
            fp += 1
        elif is_fn(prediction, y_true):
            fn += 1

    accuracy = 1.0 - (wrong_predictions / len(test_data))
    precision = p(tp, fp)
    recall = r(tp, fn)
    f1 = f(precision, recall)

    print('*******************************')
    print('****** {} ******'.format(model))
    print('*******************************')
    if hasattr(model, '_lr'):
        print('Best Learning Rate: {}'.format(model._lr))
    if hasattr(model, '_c'):
        print('Best Regularization: {}'.format(model._c))
    if hasattr(model, '_var'):
        print('Best Tradeoff: {}'.format(model._var))
    if hasattr(model, '_s'):
        print('Best Smoothing: {}'.format(model._s))
    print('Best Cross-Val F1: {}'.format(model._cross_val_f))
    print('Best Cross-Val Precision: {}'.format(model._cross_val_p))
    print('Best Cross-Val Recall: {}'.format(model._cross_val_r))
    print('Test set accuracy: {}'.format(accuracy))
    print('F1: {}'.format(f1))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('*******************************\n')

    return accuracy
