from sklearn.tree import DecisionTreeRegressor
from .utils import regressor_training


class DecisionTreeRegressionTraining():
    """
    - Initialize the Decision Tree algorithm
    - Fit the Decision Tree regressor
    - Predict the performance of the algorithm on the training and cv set


    Parameters
    ===========

        criterion: str
            The function to measure the quality of a split.

        max_tree_depth: int
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.

        seed: int
            Use to initialize the random_state of DecisionTreeRegressor which controls the randomness of the estimator.
            Important to ensure reproducibility

        input: :py:class:`tuple` of 3 elements
            There is one element for each train, cv and test set. Each element corresponds to a 2D numpy.array of
            size ``N_SAMPLES`` x ``N_PARAMETERS`` where ``N_SAMPLES`` changes in function of the set but
            ``N_PARAMETERS`` is always the same ``N_PARAMETERS`` = ``N_CONT_PARAMETERS`` + ``N_CAT_PARAMETERS``
            (after one-hot-encoding)

        target: :py:class:`tuple` of 3 elements
            Each element (1D numpy.array) corresponds to the target values for training, cv and testing set.

    Returns
    ========

        regressor: self: DecisionTreeRegressor
            Fitted estimator

        y_pred_train: numpy.array
            Array containing the predicted value for each input samples of the training set

        y_pred_cv: numpy.array
            Array containing the predicted value for each input samples of the cv set
    """

    def __init__(self, criterion, max_tree_depth, seed, input, target):
        self.criterion = criterion
        self.max_tree_depth = max_tree_depth
        self.dt_seed = seed

        self.target_train = target[0]
        self.target_cv = target[1]
        self.x_train = input[0]
        self.x_cv = input[1]

    def __call__(self):
        regressor = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_tree_depth,
                                          random_state=self.dt_seed)

        return regressor_training(regressor, self.x_train, self.x_cv, self.target_train, self.target_cv, self.criterion)
