from py_scripts.HousePricesDatabase import HousePricesDatabase
from py_scripts.DataPreprocessing import DataPreprocessing
from py_scripts.RFAlgorithm import RandomForestTraining
from py_scripts.DecisionAlgorithm import DecisionTreeRegressionTraining
from py_scripts.Analysis import Analysis
from py_scripts.utils import regressor_test


def run_toolchain(db_path, continuous_parameters, discrete_parameters, ordinal_parameters, nominal_parameters, protocol,
                  n_trees, criterion, seed, max_tree_depth_rf, max_tree_depth_dt, save_fig):
    """
    Function called in each user configuration file. It is the function which manage all the blocks and call them in a
    specific order.


    Parameters
    ===========
        db_path: str
            Path to the database csv file

        continuous_parameters: :py:class:`list` of :py:class:`str`
            list containing the name of all the continuous parameters taken into account for this experiment

        discrete_parameters: :py:class:`list` of :py:class:`str`
            list containing the name of all the discrete parameters taken into account for this experiment

        ordinal_parameters: :py:class:`list` of :py:class:`str`
            list containing the name of all the ordinal parameters taken into account for the experiment

        nominal_parameters: :py:class:`list` of :py:class:`str`
            list containing the name of all the nominal parameters taken into account for the experiment

        protocol: :py:class:`list` of :py:class:`float`
            a list of coefficients defining relative sizes of training, cv, and test sets
            Default: [0.6, 0.2, 0.2]

        n_trees: int
            The number of tree in the forest

        criterion: str
            The function to measure the quality of a split.

        seed: int
            Use to initialize the random_state of DecisionTreeRegressor which controls the randomness of the estimator.
            Important to ensure reproducibility

        max_tree_depth_rf: int
            The maximum depth of the tree in the random forest algorithm. If None, then nodes are expanded until all
            leaves are pure or until all leaves contain less than min_samples_split samples.

        max_tree_depth_dt: int
            The maximum depth of the tree in the random forest algorithm. If None, then nodes are expanded until all
            leaves are pure or until all leaves contain less than min_samples_split samples.

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

    print("1. Load Database")
    house_price_db = HousePricesDatabase(db_path, continuous_parameters, discrete_parameters, ordinal_parameters,
                                         nominal_parameters, protocol)
    train_set, cv_set, test_set = house_price_db()

    print("2. Data Preprocessing")
    preprocessing = DataPreprocessing(train_set, cv_set, test_set)
    X, y, mean_sale_price, std_sale_price = preprocessing()

    print("3. Algorithm train")
    print("-----> RF")
    rf_train = RandomForestTraining(n_trees, criterion, seed, max_tree_depth_rf, X, y)
    rf_regressor, rf_y_predict_train, rf_y_predict_cv = rf_train()

    print("-----> Decision Tree")
    decision_train = DecisionTreeRegressionTraining(criterion, seed, max_tree_depth_dt, X, y)
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
