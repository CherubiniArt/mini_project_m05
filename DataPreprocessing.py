import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessing():

    def __init__(self, train_set, cv_set, test_set):
        self.categorical_train = train_set[1]
        self.categorical_cv = cv_set[1]
        self.categorical_test = test_set[1]

        self.continuous_train = train_set[0]
        self.continuous_cv = cv_set[0]
        self.continuous_test = test_set[0]

        self.target_train = train_set[2]
        self.target_cv = cv_set[2]
        self.target_test = test_set[2]

    def __call__(self):

        # Data normalization -> continuous set only. The mean and std is computed on the training set only, to not have
        # effect of the testing set on the results
        means = np.mean(self.continuous_train, axis=0)
        stds = np.std(self.continuous_train, axis=0)

        self.continuous_train = (self.continuous_train-means)/stds
        self.continuous_cv = (self.continuous_cv - means) / stds
        self.continuous_test = (self.continuous_test - means) / stds

        # SalePrice normalization:
        mean_sale_price = np.mean(self.target_train)
        std_sale_price = np.std(self.target_train)

        self.target_train = (self.target_train - mean_sale_price)/std_sale_price
        self.target_cv = (self.target_cv - mean_sale_price) / std_sale_price
        self.target_test = (self.target_test - mean_sale_price) / std_sale_price

        # One-hot-encoder for the categorical sets:
        if self.categorical_train.shape[1] != 0:
            one_hot_encoder = OneHotEncoder(sparse=False)
            one_hot_encoder.fit(np.concatenate((self.categorical_train, self.categorical_cv, self.categorical_test), axis=0))

            self.categorical_train = one_hot_encoder.transform(self.categorical_train)
            self.categorical_cv = one_hot_encoder.transform(self.categorical_cv)
            self.categorical_test = one_hot_encoder.transform(self.categorical_test)

        # Merge the continuous and categorical sets in one array per set (train, cv, test)
        X_train = np.concatenate((self.continuous_train, self.categorical_train), axis=1)
        X_cv = np.concatenate((self.continuous_cv, self.categorical_cv), axis=1)
        X_test = np.concatenate((self.continuous_test, self.categorical_test), axis=1)

        y_train = self.target_train
        y_cv = self.target_cv
        y_test = self.target_test

        assert(len(y_train) == X_train.shape[0])
        assert (len(y_cv) == X_cv.shape[0])
        assert (len(y_test) == X_test.shape[0])

        return (X_train, X_cv, X_test), (y_train, y_cv, y_test)