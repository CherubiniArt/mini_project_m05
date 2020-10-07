from sklearn.ensemble import RandomForestRegressor
from .utils import regressor_training

class RandomForestTraining():
    """
    - Initialize the Random Forest algorithm
    - Fit the Random Forest regressor
    - Predict the performance of the algorithm on the training and cv set


    Parameters
    ===========

        n_trees: int
            The number of tree in the forest

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

        regressor: self: RandomForestRegressor
            Fitted estimator

        y_pred_train: numpy.array
            Array containing the predicted value for each input samples of the training set

        y_pred_cv: numpy.array
            Array containing the predicted value for each input samples of the cv set
    """

    def __init__(self, n_trees, criterion, max_tree_depth, rf_seed, input, target):
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_tree_depth = max_tree_depth
        self.rf_seed = rf_seed

        self.target_train = target[0]
        self.target_cv = target[1]
        self.x_train = input[0]
        self.x_cv = input[1]

    def __call__(self):
        regressor = RandomForestRegressor(n_estimators=self.n_trees, criterion=self.criterion,
                                          max_depth=self.max_tree_depth, random_state=self.rf_seed)

        return regressor_training(regressor, self.x_train, self.x_cv, self.target_train, self.target_cv, self.criterion)
