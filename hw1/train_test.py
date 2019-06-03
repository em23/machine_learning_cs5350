import numpy as np

import id3 as id3
from data import Data

DATA_DIR = 'data/'


def get_data_obj(filename):
    data = np.loadtxt(DATA_DIR + filename + '.csv', delimiter=',', dtype=str)
    return Data(data=data)


if __name__ == '__main__':

    print("\nFull Decision Tree: ")
    data_obj = get_data_obj('train')
    id3_tree = id3.id3(data_obj, data_obj.attributes, data_obj.get_column('label'))

    error, depth = id3.report_error(data_obj, id3_tree)
    print("Accuracy on training data: {}%; Depth: {}".format(100-error, depth))

    data_obj_test = get_data_obj('test')

    error, depth = id3.report_error(data_obj_test, id3_tree)
    print("    Accuracy on test data: {}%; Depth: {}".format(100-error, depth))

    print("\nTree with Max Depth 5")

    max_depth = 5
    pruned_tree = id3.pruning_tree(id3_tree, max_depth)

    error, depth = id3.report_error(data_obj_test, pruned_tree)
    print("    Accuracy on test data: {}%; Depth: {}".format(100 - error, depth))

    # x = id3.group_label(data_obj)

    # for attribute in data_obj.attributes.keys():
    #     y = id3.group_attribute_by_label(data_obj, data_obj.get_column([attribute, 'label']))
    #     print('{} = > {}'.format(attribute, id3.attribute_expected_entropy(x, y)))
    #
    # for attribute in data_obj.attributes.keys():
    #     y = id3.group_attribute_by_label(data_obj, data_obj.get_column([attribute, 'label']))
    #     print('{} = > {}'.format(attribute, id3.gain(x, y)))
