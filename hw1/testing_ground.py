import unittest

from data import Data
from id3 import id3, report_error, gain, group_attribute_by_label, group_label, attribute_expected_entropy
import numpy as np

DATA_DIR = 'data/'


class Test_id3_on_Tennis(unittest.TestCase):

    def setUp(self):
        data = np.loadtxt(DATA_DIR + 'tennis.csv', delimiter=' ', dtype=str)
        self.data_obj = Data(data=data)
        self.id3_tree = id3(self.data_obj, self.data_obj.attributes, self.data_obj.get_column('label'))

    def test_gain(self):
        x = group_label(self.data_obj)

        y = group_attribute_by_label(self.data_obj, self.data_obj.get_column(['O', 'label']))
        self.assertEqual('0.246', "%.3f" % (gain(x, y)-0.0005))

        y = group_attribute_by_label(self.data_obj, self.data_obj.get_column(['T', 'label']))
        self.assertEqual('0.029', "%.3f" % (gain(x, y) - 0.0005))

        y = group_attribute_by_label(self.data_obj, self.data_obj.get_column(['H', 'label']))
        self.assertEqual('0.151', "%.3f" % (gain(x, y) - 0.0005))

        y = group_attribute_by_label(self.data_obj, self.data_obj.get_column(['W', 'label']))
        self.assertEqual('0.048', "%.3f" % (gain(x, y)-0.0005))


    def test_error_on_train(self):
        error, depth = report_error(self.data_obj, self.id3_tree)
        self.assertEqual(0.0, error)
        self.assertEqual(2, depth)


if __name__ == '__main__':
    unittest.main()


