from models.decaying_perceptron import DecayingPerceptron
from models.averaged_perceptron import AveragedPerceptron
from models.svm import SVM
from models.logistic_regression import LogisticRegression
from models.naive_bayes import NaiveBayes

import helper_methods as hm
import numpy as np
np.random.seed(1)

if __name__ == '__main__':
    ############################################
    ###### Part I                    ###########
    ############################################
    learning_rates = [1, 0.1, 0.01]

    dp = DecayingPerceptron()
    dp.train(learning_rates)
    dp.report()
    dp.evaluate()

    ap = AveragedPerceptron()
    ap.train(learning_rates)
    ap.report()
    ap.evaluate()


    ############################################
    ###### Part II                   ###########
    ############################################

    svm = SVM(verbose=True)
    svm.train(epochs=20)
    hm.report(svm)
    hm.evaluate(svm)

    lr = LogisticRegression(verbose=True)
    lr.train(epochs=20)
    hm.report(lr)
    hm.evaluate(lr)

    nb = NaiveBayes()
    nb.train(epochs=1)
    hm.report(nb)
    hm.evaluate(nb)

    # Logistic regression using sklearn
    import data as dt
    from sklearn.linear_model import LogisticRegression

    train_data = dt.load_data(dt.TRAIN, matrix=True)
    test_data = dt.load_data(dt.TEST, matrix=True)

    lr = LogisticRegression()
    lr.fit(X=train_data[:, 1:], y=train_data[:, 0])

    print('************************************************')
    print('****** Logistic Regression (scikit-learn) ******')
    print('************************************************')
    print("Parameters: {}".format(lr.get_params()))
    print('Score: {}'.format(lr.score(X=test_data[:, 1:], y=test_data[:, 0])))
    print('************************************************\n')

    test_data = dt.load_data(dt.EVAL, matrix=True)
    test_data_id = dt.np_load_data(dt.EVAL_ID)

    import csv
    with open('eval.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, quotechar=' ')
        spamwriter.writerow(["example_id", "label"])

        pred_eval = lr.predict(test_data[:, 1:])
        E = len(pred_eval)
        for i in range(E):
            if pred_eval[i] <= 0:
                spamwriter.writerow([test_data_id[i], "0"])
            else:
                spamwriter.writerow([test_data_id[i], '1'])

