from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np


class Analysis():
    """
    Used to compute and plot the performance of the experiment


    Parameters
    ===========

        y_pred_train: list
            The first element of the list is an array containing the predicted values for each input samples of the
            training set computed with the RandomForestRegressor, while the second element is the predicted values
            computed with DecisionTreeRegressor.

        y_pred_cv: list
            The first element of the list is an array containing the predicted values for each input samples of the
            cv set computed with the RandomForestRegressor, while the second element is the predicted values
            computed with DecisionTreeRegressor.

        y_pred_test: list
            The first element of the list is an array containing the predicted values for each input samples of the
            testing set computed with the RandomForestRegressor, while the second element is the predicted values
            computed with DecisionTreeRegressor.

        target: :py:class:`tuple` of 3 elements
            Each element (1D numpy.array) corresponds to the target values for training, cv and testing set.

        mean_target: float
            Mean value used to do the z-normalization of the target values

        std_target: float
            Standard deviation value used to do the z-normalization of the target values

        algorithm: ["RF", "Decision Tree"]
            Contains the name of the algorithms tested in the experiments.

        save_fig: str
            Indicate the path where to save the figure with the results

    Returns
    =======

        r2_train: :py:class:`list` of :py:class:`float`
            Contains the R2 score computed on the training set for both Decison Tree and Random Forest algorithm

        r2_cv: :py:class:`list` of :py:class:`str`
            Contains the R2 score computed on the cv set for both Decison Tree and Random Forest algorithm

        r2_test: :py:class:`list` of :py:class:`str`
            Contains the R2 score computed on the test set for both Decison Tree and Random Forest algorithm
    """

    def __init__(self, y_pred_train, y_pred_cv, y_pred_test, target, mean_target, std_target, algorithm, save_fig):
        self.y_pred_train = y_pred_train
        self.y_pred_cv = y_pred_cv
        self.y_pred_test = y_pred_test

        self.y_train = target[0]
        self.y_cv = target[1]
        self.y_test = target[2]

        self.mean = mean_target
        self.std = std_target

        self.algorithm = algorithm

        self.save_fig = save_fig

    def autolabel(self, ax, rects):
        # Indicate the score of each bar in the plot bar
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.0 * h, '%f' % h,
                    ha='center', va='bottom')

    def plot_multiple_bars(self, scores_train, scores_cv, scores_test):
        # Number of compared algorithms
        N = len(self.algorithm)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.27  # the width of the bars

        fig = plt.figure()
        ax = fig.add_subplot(111)

        rects1 = ax.bar(ind, scores_train, width, color='r')
        rects2 = ax.bar(ind + width, scores_cv, width, color='g')
        rects3 = ax.bar(ind + width * 2, scores_test, width, color='b')

        ax.set_ylabel('R2 Scores')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(self.algorithm)
        ax.legend((rects1[0], rects2[0], rects3[0]), ('train', 'cv', 'test'))

        self.autolabel(ax, rects1)
        self.autolabel(ax, rects2)
        self.autolabel(ax, rects3)

        if self.save_fig is not None:
            plt.savefig(self.save_fig)
            plt.show()

    def __call__(self):
        r2_train = []
        r2_cv = []
        r2_test = []

        for i, alg in enumerate(self.algorithm):
            r2_train.append(r2_score(self.y_train, self.y_pred_train[i]))
            r2_cv.append(r2_score(self.y_cv, self.y_pred_cv[i]))
            r2_test.append(r2_score(self.y_test, self.y_pred_test[i]))

            print(alg+":")
            print("r2 train =", r2_train[-1])
            print("r2 cv =", r2_cv[-1])
            print("r2 test =", r2_test[-1])

        self.plot_multiple_bars(r2_train, r2_cv, r2_test)

        return r2_train, r2_cv, r2_test
