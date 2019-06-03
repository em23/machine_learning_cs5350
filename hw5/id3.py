from statistics import mode, StatisticsError
from copy import deepcopy
from scipy.stats import entropy
import numpy as np


class Node:
    def __init__(self, attribute_name: str, depth=0):
        self._depth = depth
        self._children = {}
        self._attribute_name = attribute_name

    def addChild(self, attribute_value, child = None):
        self._children[attribute_value] = child

    def get_attribute_name(self):
        return self._attribute_name

    def get_children(self):
        return self._children

    def set_depth(self, depth):
        self._depth = depth

    def get_depth(self):
        return self._depth


def id3(data_obj: object, attributes: dict, labels: list, depth= -1):
    label = labels[0]
    same_label = True

    for i in range(len(labels)):
        if labels[i] != label:
            same_label = False
            break

    if same_label:
        return Node(label, depth + 1)

    attribute_name = get_best_attribute(data_obj)

    if depth >= 20:
        temp_obj = data_obj.get_row_subset(attribute_name, 1, data_obj.raw_data)
        try:
            common_value = mode(temp_obj.get_column('label', temp_obj.raw_data))
        except StatisticsError:
            common_value = label
        return Node(common_value, depth+1)

    root = Node(attribute_name, depth+1)


    for v in data_obj.get_attribute_possible_vals(attribute_name):
        new_data_obj = data_obj.get_row_subset(attribute_name, v, data_obj.raw_data)

        if len(new_data_obj) == 0:
            try:
                common_value = mode(data_obj.get_column('label', data_obj.raw_data))
            except StatisticsError:
                common_value = label
            root.addChild(v, Node(common_value, depth + 2))
        else:
            new_attributes = deepcopy(attributes)
            if attribute_name in new_attributes:
                new_attributes.pop(attribute_name)
            root.addChild(v, id3(new_data_obj, new_attributes, new_data_obj.get_column('label'), depth+1))

    return root


def pruning_tree(id3_tree, max_depth):
    root = id3_tree
    if len(id3_tree.get_children()) == 0 :
        return id3_tree

    pruned_tree = explore_tree(root, max_depth)

    return pruned_tree


def explore_tree(node, max_depth):
    current_depth = node.get_depth()

    if current_depth >= max_depth:
        label_values = []
        get_label_values(node, label_values)

        try:
            common_value = mode(label_values)
        except StatisticsError:
            common_value = label_values[0]

        return Node(common_value, depth= current_depth)

    for attribute_value, child in node.get_children().items():
        node.addChild(attribute_value, explore_tree(child, max_depth))

    return node


def get_label_values(node, label_values):
    if len(node.get_children()) == 0:
        label_values.append(node.get_attribute_name())
        return

    for attribute_value, child in node.get_children().items():
        get_label_values(child, label_values)



"""
    Gets a Data object 
    
    Returns a key:value dict for label:total number 
"""
def group_label(data_obj):
    possible_label_vals = np.unique(data_obj.get_column('label'))
    label_count = dict(zip(possible_label_vals, [0] * len(possible_label_vals)))

    for label in data_obj.get_column('label'):
        label_count[label] += 1

    return label_count


"""
    Gets a Data object and a list of tuples for [[attribute, label]...]
    
    Returns a key:[key:value] dict for attribute:[label:total number, ...] 
"""
def group_attribute_by_label(data_obj, attribute_label_cols):
    attribute_grouped_by_label = {}

    for attribute, label in attribute_label_cols:

        if attribute not in attribute_grouped_by_label.keys():
            possible_label_vals = np.unique(data_obj.get_column('label'))
            label_data = dict(zip(possible_label_vals, [0] * len(possible_label_vals)))

            attribute_grouped_by_label[attribute] = label_data

        attribute_grouped_by_label[attribute][label] += 1

    return attribute_grouped_by_label


"""
    Gets a {key:value} dict for {label:total number} and {attribute:{label:total number, ...}}
    
    Returns Expected Information Gain for specific attribute 
"""
def gain(label_count, attribute_grouped_by_label):
    total_entropy = entropy([x/sum(label_count.values()) for x in label_count.values()], base=2)
    expected_entropy = attribute_expected_entropy(label_count, attribute_grouped_by_label)

    return total_entropy - expected_entropy


def attribute_expected_entropy(label_count, attribute_grouped_by_label):
    attribute_entropy = []

    for attribute_value in attribute_grouped_by_label.items():
        fraction = sum(attribute_value[1].values()) / sum(label_count.values())
        attribute_value_entropy = fraction * entropy(
            [x / sum(attribute_value[1].values()) for x in attribute_value[1].values()], base=2)

        attribute_entropy.append(attribute_value_entropy)

    return sum(attribute_entropy)


def get_best_attribute(data_obj):
    max_gain = ('', 0.0)

    for i in data_obj.attributes.keys():
        attribute_label_cols = data_obj.get_column([i, 'label'])
        attribute_grouped_by_label = group_attribute_by_label(data_obj, attribute_label_cols)

        current_gain = gain(group_label(data_obj), attribute_grouped_by_label)
        if current_gain >= max_gain[1]:
            max_gain = (i, current_gain)

    return max_gain[0]


def report_error(data_obj, main_root):
    max_depth = 0
    wrong_predictions = 0

    for test in data_obj.raw_data:
        root = main_root

        while len(root.get_children()) != 0:
            current_attribute = root.get_attribute_name()
            attribute_index = data_obj.get_column_index(current_attribute)
            next_attribute_value = test[attribute_index]

            if next_attribute_value in root.get_children().keys():
                root = root.get_children()[next_attribute_value]
            else:
                try:
                    common_value = mode(data_obj.get_column('label', data_obj.raw_data))
                except StatisticsError:
                    common_value = data_obj.get_column('label', data_obj.raw_data)[0]

                root = Node(common_value, depth= root.get_depth() + 1)
                if max_depth < root.get_depth():
                    max_depth = root.get_depth()
                break

            if max_depth < root.get_depth():
                max_depth = root.get_depth()

        attribute_index = data_obj.get_column_index('label')
        if test[attribute_index] != root.get_attribute_name():
            wrong_predictions += 1

    training_error = wrong_predictions/len(data_obj.raw_data)*100

    return training_error, max_depth


def predict(data_obj, example, id3_tree_root):
    root = id3_tree_root
    test = example

    while len(root.get_children()) != 0:
        current_attribute = root.get_attribute_name()
        attribute_index = data_obj.get_column_index(current_attribute)
        next_attribute_value = test[attribute_index]

        if next_attribute_value in root.get_children().keys():
            root = root.get_children()[next_attribute_value]
        else:
            try:
                common_value = mode(data_obj.get_column('label', data_obj.raw_data))
            except StatisticsError:
                common_value = data_obj.get_column('label', data_obj.raw_data)[0]
            root = Node(common_value, depth=root.get_depth() + 1)
            break

        attribute_index = data_obj.get_column_index('label')

        # return test[attribute_index] != root.get_attribute_name()
    return root.get_attribute_name()