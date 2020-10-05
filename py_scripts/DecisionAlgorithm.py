from sklearn.tree import DecisionTreeRegressor
from py_scripts.utils import regressor_training
import random


class DecisionTreeRegressionTraining():

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
