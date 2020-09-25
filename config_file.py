from HousePricesDatabase import HousePricesDatabase
from DataPreprocessing import DataPreprocessing
from RFAlgorithm import RandomForestTraining, RandomForestTesting
from Analysis import Analysis

# ======================================================================================================================
# Defined by the users:
# Path to the db used for the regression task
db_path = "house-prices/house-prices.csv"

# Parameters used for the regression task
continuous_parameters = ["Gr Liv Area", "Garage Area", "Total Bsmt SF", "1st Flr SF", "Mas Vnr Area"]
discrete_parameters = ["Year Built", "Year Remod/Add", "Full Bath", "TotRms AbvGrd", "Fireplaces"]
ordinal_parameters = []  # ["Garage Type", "Bsmt Qual"]
nominal_parameters = []

# Protocol used to split the dataset into train/cv/test
protocol = [0.8, 0.1, 0.1]

# Decision trees details:
n_trees = 50
criterion = "mse"
max_tree_depth = 10
rf_seed = 10  # Used to fix the random_state of RFregressor to ensure reproducibility

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

print("-----> Decision Tree")
# Put here your code :)

print("4. RF test")
print("-----> RF")
regressor_testing = RandomForestTesting(regressor, criterion, X, y)
y_predict_test = regressor_testing()

print("-----> Decision Tree")
# Put here your code :)

# 5. Analysis
print("5. Analysis")
score = Analysis((y_predict_train, y_predict_cv, y_predict_test), y, mean_sale_price, std_sale_price)
score()

# To adapt
