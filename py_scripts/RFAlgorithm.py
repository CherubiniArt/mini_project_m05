from sklearn.ensemble import RandomForestRegressor
from py_scripts.utils import regressor_training


class RandomForestTraining():

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

        return regressor_training(regressor, self.x_train, self.x_cv, self.target_train, self.target_cv,
                                    self.criterion)
