from sklearn.ensemble import RandomForestRegressor
import numpy as np


def mse(y_pred, y_target):
    return np.mean((y_pred - y_target) ** 2)


def mae(y_pred, y_target):
    return np.mean(abs(y_pred - y_target))


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

        regressor.fit(self.x_train, self.target_train)

        # Predict the cv set
        y_pred_train = regressor.predict(self.x_train)
        y_pred_cv = regressor.predict(self.x_cv)

        if self.criterion == "mse":
            criterion_train = mse(y_pred_train, self.target_train)
            criterion_cv = mse(y_pred_cv, self.target_cv)
        elif self.criterion == "mae":
            criterion_train = mae(y_pred_train, self.target_train)
            criterion_cv = mae(y_pred_cv, self.target_cv)

        print(self.criterion + " train =", criterion_train)
        print(self.criterion + " cv =", criterion_cv)

        return regressor, y_pred_train, y_pred_cv


class RandomForestTesting():

    def __init__(self, model, criterion, input, target):
        self.model = model
        self.criterion = criterion
        self.target_test = target[2]
        self.x_test = input[2]

    def __call__(self):
        # Predict the test set
        y_pred_test = self.model.predict(self.x_test)

        if self.criterion == "mse":
            criterion_test = mse(y_pred_test, self.target_test)
        elif self.criterion == "mae":
            criterion_test = mae(y_pred_test, self.target_test)

        print(self.criterion + " test =", criterion_test)

        return y_pred_test
