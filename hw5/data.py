import numpy as np

DATA_DIR = 'data/'
TRAIN = 'train.liblinear'
TEST = 'test.liblinear'
FOLDS = ['CVSplits/training0' + str(x) + '.data' for x in range(5)]


def load_data(fname):
    file_path = DATA_DIR + fname

    nr_of_examples = _file_len(file_path)
    data = np.zeros(shape=(nr_of_examples, 220))

    train_file = open(file_path, mode='r')
    for index, line in enumerate(train_file):
        line_columns = line.split(' ')

        label = int(line_columns[0])
        features = line_columns[1:]

        data[index, 0] += label

        for feature in features:
            column, value = feature.split(':')
            data[index, int(column)] += int(value)

    return data


def _file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == "__main__":
    train_data = load_data(TRAIN)
    test_data = load_data(TEST)

    print(train_data)
    print(test_data)

    # labels = train_data[:, 0]
    # import statistics as st
    # y = st.mode(labels)
    #
    # correct_predictions = 0
    # for x in test_data[:, 0]:
    #     if x == y:
    #         correct_predictions += 1
    #
    # print(correct_predictions / len(train_data[:, 0]))