import numpy as np
import statistics as st

import id3 as id3
from data import Data

DATA_DIR = 'data/CVfolds_new/'


if __name__ == '__main__':
    filenames = ['fold' + str(x) for x in range(1,6)]
    depths = [1, 2, 3, 4, 5, 10, 15]

    train_objs = []
    test_objs = []

    datas = []
    for i in range(len(filenames)):
        for index, filename in enumerate(filenames):
            if i == index:
                test_data = np.loadtxt(DATA_DIR + filename + '.csv', delimiter=',', dtype=str)
                test_obj = Data(data=test_data)
                test_objs.append(test_obj)
                continue

            datas.append(np.loadtxt(DATA_DIR + filename + '.csv', delimiter=',', dtype=str))

        data = np.concatenate(datas)
        data_obj = Data(data=data)
        train_objs.append(data_obj)

    avg_accuracies = []
    for max_depth in depths:
        accuracies = []
        print('**********************************')
        print('****** Hyperparameter is {} ******'.format(max_depth))
        print('**********************************\n')

        for i in range(len(filenames)):
            id3_tree = id3.id3(train_objs[i], train_objs[i].attributes, train_objs[i].get_column('label'))
            pruned_tree = id3.pruning_tree(id3_tree, max_depth)

            error, depth = id3.report_error(test_objs[i], pruned_tree)
            accuracies.append(100.0-error)

            print('***** Testing on {} *****'.format(filenames[i]))

        avg_accuracy = st.mean(accuracies)
        avg_accuracies.append(avg_accuracy)
        print("Average accuracy: {}%; Standard Deviation: {}\n".format(avg_accuracy, np.std(accuracies)))

    # print(dict(zip(depths, avg_accuracies)))

