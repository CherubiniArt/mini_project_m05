from py_scripts.HousePricesDatabase import HousePricesDatabase
from py_scripts.DataPreprocessing import DataPreprocessing
from py_scripts.RFAlgorithm import RandomForestTraining
from py_scripts.DecisionAlgorithm import DecisionTreeRegressionTraining
from py_scripts.Analysis import Analysis
from py_scripts.utils import regressor_test

import numpy as np


def run_toolchain(db_path, continuous_parameters, discrete_parameters, ordinal_parameters, nominal_parameters, protocol,
                  n_trees, criterion, rf_seed, max_tree_depth_rf, max_tree_depth_dt, save_fig):

    print("1. Load Database")
    house_price_db = HousePricesDatabase(db_path, continuous_parameters, discrete_parameters, ordinal_parameters,
                                         nominal_parameters, protocol)
    train_set, cv_set, test_set = house_price_db()

    print("2. Data Preprocessing")
    preprocessing = DataPreprocessing(train_set, cv_set, test_set)
    X, y, mean_sale_price, std_sale_price = preprocessing()

    print("3. Algorithm train")
    print("-----> RF")
    rf_train = RandomForestTraining(n_trees, criterion, rf_seed, max_tree_depth_rf, X, y)
    rf_regressor, rf_y_predict_train, rf_y_predict_cv = rf_train()

    print("-----> Decision Tree")
    # Put here your code :)
    decision_train = DecisionTreeRegressionTraining(criterion, rf_seed, max_tree_depth_dt, X, y)
    dt_regressor, dt_y_predict_train, dt_y_predict_cv = decision_train()

    print("5. Algorithm testing")
    print("-----> RF")
    rf_y_predict_test = regressor_test(rf_regressor, X[2], y[2], criterion)

    print("-----> Decision Tree")
    dt_y_predict_test = regressor_test(dt_regressor, X[2], y[2], criterion)

    print("6. Analysis")
    y_pred_train = [rf_y_predict_train, dt_y_predict_train]
    y_pred_cv = [rf_y_predict_cv, dt_y_predict_cv]
    y_pred_test = [rf_y_predict_test, dt_y_predict_test]
    algorithm = ["RF", "Decision Tree"]

    score = Analysis(y_pred_train, y_pred_cv, y_pred_test, y, mean_sale_price, std_sale_price, algorithm, save_fig)
    r2_train, r2_cv, r2_test = score()

    return r2_train, r2_cv, r2_test
