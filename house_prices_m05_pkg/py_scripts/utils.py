import numpy as np
import nose


def mse(y_pred, y_target):
    return np.mean((y_pred - y_target) ** 2)


def mae(y_pred, y_target):
    return np.mean(abs(y_pred - y_target))


def regressor_training(regressor, x_train, x_cv, target_train, target_cv, criterion):
    """
    Function used to fit/train a specific regressor and compute its performance on the training and cv set

    Parameters
    ===========

        regressor: RandomForestRegressor or DecisionTreeRegressor
            Initialized regressor

        x_train: numpy.array
            Array containing the samples used for the training of the regressor. The size is
            ``N_SAMPLES_TRAIN`` x ``N_PARAMETERS``

        x_cv: numpy.array
            Array containing the samples used for the training of the regressor. The size is
            ``N_SAMPLES_CV`` x ``N_PARAMETERS``

        target_train: numpy.array
            Each element of the array corresponds to the target values in the training set

        target_cv: numpy.array
            Each element of the array corresponds to the target values in the cv set

        criterion: str
            The function to measure the quality of a split.

    Returns
    ========
        regressor: self.DecisionTreeRegressor or self.RandomForestRegressor
            Fitted estimator

        y_pred_train: numpy.array
            Array containing the predicted value for each input samples of the training set

        y_pred_cv: numpy.array
            Array containing the predicted value for each input samples of the cv set

    """
    regressor.fit(x_train, target_train)

    # Predict the cv set
    y_pred_train = regressor.predict(x_train)
    y_pred_cv = regressor.predict(x_cv)

    if criterion == "mse":
        criterion_train = mse(y_pred_train, target_train)
        criterion_cv = mse(y_pred_cv, target_cv)
    elif criterion == "mae":
        criterion_train = mae(y_pred_train, target_train)
        criterion_cv = mae(y_pred_cv, target_cv)

    print(criterion + " train =", criterion_train)
    print(criterion + " cv =", criterion_cv)

    return regressor, y_pred_train, y_pred_cv


@nose.tools.nottest
def regressor_test(model, x_test, target_test, criterion):
    """
    Function used to predict the performance on the testing set of a specific regressor

    Parameters
    ===========

        model: RandomForestRegressor or DecisionTreeRegressor
            Fitted regressor

        x_test: numpy.array
            Array containing the samples used for the testing of the regressor. The size is
            ``N_SAMPLES_TEST`` x ``N_PARAMETERS``

        target_test: numpy.array
            Each element of the array corresponds to the target values in the testing set

        criterion: str
            The function to measure the quality of a split.

    Returns
    ========

        y_pred_test: numpy.array
            Array containing the predicted value for each input samples of the test set computed with model

    """
    y_pred_test = model.predict(x_test)

    if criterion == "mse":
        criterion_test = mse(y_pred_test, target_test)
    elif criterion == "mae":
        criterion_test = mae(y_pred_test, target_test)

    print(criterion + " test =", criterion_test)

    return y_pred_test
