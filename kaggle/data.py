import numpy as np

DATA_DIR = 'movie-ratings/data-splits/'

TRAIN = 'data.train'
EVAL = 'data.eval.anon'
TEST = 'data.test'

TRAIN_ID = 'data.train.id'
EVAL_ID = 'data.eval.anon.id'
TEST_ID = 'data.test.id'

FOLDS = ['CVSplits/training0' + str(x) + '.data' for x in range(5)]


def load_data(fname, matrix=False):
    file_path = DATA_DIR + fname

    nr_of_examples = _file_len(file_path)
    if matrix:
        data = np.zeros(shape=(nr_of_examples, 74482))
    else:
        data = [None]*nr_of_examples

    train_file = open(file_path, mode='r')
    for index, line in enumerate(train_file):
        line_columns = line.split(' ')

        label = 1 if int(line_columns[0]) == 1 else -1
        features = line_columns[1:]

        if matrix:
            data[index, 0] += label
        else:
            feature_parsed = {}

        for feature in features:
            column, value = feature.split(':')
            if matrix:
                data[index, int(column)] += float(value)
            else:
                feature_parsed[int(column)] = int(value)

        if not matrix:
            data[index] = (label, feature_parsed)

    return data


def np_load_data(fname):
    file_path = DATA_DIR + fname
    data = np.loadtxt(file_path, delimiter=',', dtype=str)
    return data

def _file_len(fname):
    with open(fname) as f:

        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == "__main__":
    train_data = load_data(TRAIN)
    dev_data = np_load_data(EVAL_ID)

