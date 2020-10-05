import numpy as np
import nose


def mse(y_pred, y_target):
    return np.mean((y_pred - y_target) ** 2)


def mae(y_pred, y_target):
    return np.mean(abs(y_pred - y_target))


def regressor_training(regressor, x_train, x_cv, target_train, target_cv, criterion):
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
    y_pred_test = model.predict(x_test)

    if criterion == "mse":
        criterion_test = mse(y_pred_test, target_test)
    elif criterion == "mae":
        criterion_test = mae(y_pred_test, target_test)

    print(criterion + " test =", criterion_test)

    return y_pred_test
