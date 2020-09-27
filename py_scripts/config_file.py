from HousePricesDatabase import HousePricesDatabase
from DataPreprocessing import DataPreprocessing
from RFAlgorithm import RandomForestTraining, RandomForestTesting
from DecisionAlgorithm import DecisionTreeRegressionTraining, DecisionTreeRegressionTesting
from Analysis import Analysis

import numpy as np
# ======================================================================================================================
""" All the parameters that can be chosen
# Nominal parameters:
nominal = ["MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood",
           "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st",
           "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", "Misc Feature",
           "Sale Type", "Sale Condition"]

# Continuous parameters:
continuous = ["Lot Frontage", "Lot Area", "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
             "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area", "Garage Area",
             "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area",
             "Misc Val"]

# Ordinal parameters:
ordinal = ["Lot Shape", "Utilities", "Land Slope", "Overall Qual", "Overall Cond", "Exter Qual", "Exter Cond",
           "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2", "Heating QC", "Electrical",
           "Kitchen Qual", "Functional", "Fireplace Qu", "Garage Finish", "Garage Qual", "Garage Cond",
           "Paved Drive", "Pool QC", "Fence"]

# Discrete parameters:
discrete = ["Year Built", "Year Remod/Add", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath",
            "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces", "Garage Yr Blt", "Garage Cars",
            "Mo Sold", "Yr Sold"]
"""

# Defined by the users:
# Path to the db used for the regression task
db_path = "../house-prices/house-prices.csv"

# Parameters used for the regression task
"""continuous_parameters = ["Gr Liv Area", "Garage Area", "Total Bsmt SF", "1st Flr SF", "Mas Vnr Area"]
discrete_parameters = ["Year Built", "Year Remod/Add", "Full Bath", "TotRms AbvGrd", "Fireplaces"]
ordinal_parameters = []  # ["Garage Type", "Bsmt Qual"]
nominal_parameters = []"""

continuous_parameters = ["Lot Frontage", "Lot Area", "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
             "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area", "Garage Area",
             "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area",
             "Misc Val"]
discrete_parameters = ["Year Built", "Year Remod/Add", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath",
            "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces", "Garage Yr Blt", "Garage Cars",
            "Mo Sold", "Yr Sold"]
ordinal_parameters = ["Lot Shape", "Utilities", "Land Slope", "Overall Qual", "Overall Cond", "Exter Qual", "Exter Cond",
           "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2", "Heating QC", "Electrical",
           "Kitchen Qual", "Functional", "Fireplace Qu", "Garage Finish", "Garage Qual", "Garage Cond",
           "Paved Drive", "Pool QC", "Fence"]
nominal_parameters = ["MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood",
           "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st",
           "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", "Misc Feature",
           "Sale Type", "Sale Condition"]

# Protocol used to split the dataset into train/cv/test
protocol = [0.8, 0.1, 0.1]

# Decision trees details:
n_trees = 50
criterion = "mse"
max_tree_depth = 10
rf_seed = 10  # Used to fix the random_state of RF and decision tree regressors to ensure reproducibility
flag = "" # to distinguish random forest to decision tree algorithm
# ======================================================================================================================
# Toolchain
print("1. Load Database")
house_price_db = HousePricesDatabase(db_path, continuous_parameters, discrete_parameters, ordinal_parameters,
                                     nominal_parameters, protocol)
train_set, cv_set, test_set = house_price_db()

print("2. Data Preprocessing")
preprocessing = DataPreprocessing(train_set, cv_set, test_set)
X, y, mean_sale_price, std_sale_price = preprocessing()

print("3. Algorithm train")
print("-----> RF")
rf_train = RandomForestTraining(n_trees, criterion, rf_seed, max_tree_depth, X, y)
regressor, y_predict_train, y_predict_cv = rf_train()

print("4.-----> Decision Tree")
# Put here your code :)
decision_train = DecisionTreeRegressionTraining(criterion, rf_seed, max_tree_depth, X, y)
d_regressor, d_y_predict_train, d_y_predict_cv = decision_train()

print("5. RF test")
print("-----> RF")
regressor_testing = RandomForestTesting(regressor, criterion, X, y)
y_predict_test = regressor_testing()

print("6.-----> Decision Tree test")
decision_regressor_testing = DecisionTreeRegressionTesting(d_regressor, criterion, X, y)
d_y_predict_test = decision_regressor_testing()

# 5. Analysis
print("7. Analysis with Random Forest")
score = Analysis((y_predict_train, y_predict_cv, y_predict_test), y, mean_sale_price, std_sale_price, flag = "RF")
score()

print("8. Analysis with Decision tree")
score = Analysis((d_y_predict_train, d_y_predict_cv, d_y_predict_test), y, mean_sale_price, std_sale_price, flag = "DT")
score()
# To adapt
