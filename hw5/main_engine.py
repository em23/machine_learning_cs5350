import id3
import data as dt
import numpy as np
import helper_methods as hp

from id3_data import Data
from models.svm import SVM
from logistic_regression import LogisticRegression
from naive_bayes import NaiveBayes

np.random.seed(1)

if __name__ == '__main__':
    # svm = SVM()
    # svm.train(epochs=20)
    # hp.report(svm)
    #
    # lr = LogisticRegression()
    # lr.train(epochs=20)
    # hp.report(lr)

    nb = NaiveBayes()
    nb.train()
    hp.report(nb)

    # SVM Over Trees
    # print('**********************************')
    # print('****** Depth is {} ******'.format(20))
    # print('**********************************\n')
    #
    # whole_train_data = dt.load_data(dt.TRAIN)
    # whole_test_data = dt.load_data(dt.TEST)
    #
    # train_size = len(whole_train_data) // 10 # 1/10 of the whole dataset
    # test_size = len(whole_test_data)

    # building tress
    # id3_trees = []
    # number_of_trees= 200
    # for i in range(number_of_trees):
    #     np.random.shuffle(whole_train_data)
    #
    #     train_data = Data(data=whole_train_data[:train_size])
    #
    #     id3_tree = id3.id3(train_data, train_data.attributes, train_data.get_column('label'))
    #     id3_trees.append(id3_tree)
    #
    #     print("{}. Tree Built".format(i))
    #
    # transformed_test_data = np.zeros((test_size, number_of_trees+1))
    # test_obj = Data(data=whole_test_data)
    #
    # # iterate every test in the test data
    # for row, test in enumerate(whole_test_data):
    #     # assign true label to new dataset
    #     transformed_test_data[row, 0] = test[0]
    #
    #     # predict a label over each built tree
    #     for col, tree in enumerate(id3_trees, 1):
    #         label = id3.predict(test_obj, test, tree)
    #         transformed_test_data[row, col] = label
    #
    # svm = SVM(D=number_of_trees)
    # svm.train(epochs=20, train_data=transformed_test_data)
    # hp.report(svm, test_data=transformed_test_data)

