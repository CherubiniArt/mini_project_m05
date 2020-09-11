from HousePricesDatabase import HousePricesDatabase
from DataPreprocessing import DataPreprocessing
from sklearn.ensemble import RandomForestRegressor

import numpy as np

# Database:

# ======================================================================================================================
# Defined by the users:
# Path to the db used for the regression task
db_path = "house-prices/house-prices.csv"

# Parameters used for the regression task
continuous_parameters = ["Gr Liv Area", "Garage Area", "Total Bsmt SF", "1st Flr SF", "Mas Vnr Area"]
discrete_parameters = ["Year Built", "Year Remod/Add", "Full Bath", "TotRms AbvGrd", "Fireplaces"]
ordinal_parameters = ["Garage Type", "Bsmt Qual"]
nominal_parameters = []

# Protocol used to split the dataset into train/cv/test
protocol = [0.6, 0.2, 0.2]  # To verify the name !

# Decision trees details:
n_trees = 1000
criterion = "mse"
max_tree_depth = 50
# ======================================================================================================================

class RandomForestTraining():

    def __init__(self, n_trees, criterion, max_tree_depth, input, target):
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_tree_depth = max_tree_depth

        self.target_train = target[0]
        self.target_cv = target[1]
        self.x_train = input[0]
        self.x_cv = input[1]


    def __call__(self):
        regressor = RandomForestRegressor(n_estimators=self.n_trees, criterion=self.criterion, max_depth=self.max_tree_depth)

        regressor.fit(self.x_train, self.target_train)

        # Predict the cv set
        y_pred = regressor.predict(self.x_cv)

        MSE = np.mean((y_pred - self.target_cv)**2)

        print("MSE_CV =", MSE)



house_price_db = HousePricesDatabase(db_path, continuous_parameters, discrete_parameters, ordinal_parameters,
                                     nominal_parameters, protocol)

train_set, cv_set, test_set = house_price_db()

preprocessing = DataPreprocessing(train_set, cv_set, test_set)
X, y = preprocessing()

rf_train = RandomForestTraining(n_trees, criterion, max_tree_depth, X, y)

rf_train()











# Data preprocessing:


# Training


# Testing


# Analysis