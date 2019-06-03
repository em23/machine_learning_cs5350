from simple_perceptron import SimplePerceptron
from decaying_perceptron import DecayingPerceptron
from margin_perceptron import MarginPerceptron
from averaged_perceptron import AveragedPerceptron

# import matplotlib.pyplot as plt

if __name__ == '__main__':
    learning_rates = [1, 0.1, 0.01]
    margin_rates = [1, 0.1, 0.01]

    sp = SimplePerceptron()
    sp.train(learning_rates)
    sp.report()
    # plt.scatter(*zip(*sp._epoch_acc))
    # plt.plot(*zip(*sp._epoch_acc))
    # plt.show()

    dp = DecayingPerceptron()
    dp.train(learning_rates)
    dp.report()
    # plt.scatter(*zip(*dp._epoch_acc))
    # plt.plot(*zip(*dp._epoch_acc))
    # plt.show()

    mp = MarginPerceptron()
    mp.train(learning_rates, margin_rates)
    mp.report()
    # plt.scatter(*zip(*mp._epoch_acc))
    # plt.plot(*zip(*mp._epoch_acc))
    # plt.show()

    ap = AveragedPerceptron()
    ap.train(learning_rates)
    ap.report()
    # plt.scatter(*zip(*ap._epoch_acc))
    # plt.plot(*zip(*ap._epoch_acc))
    # plt.show()
