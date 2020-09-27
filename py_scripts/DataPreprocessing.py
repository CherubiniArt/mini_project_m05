import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessing():

    def __init__(self, train_set, cv_set, test_set):
        """
        Attributes
        --------------
        train_set: tuple of 3 elements
            1. 2D numpy.array of size N_SAMPLES x N_CONT_PARAMS where N_CONT_PARAMS is the number of discrete/continuous
                parameters and N_SAMPLES is the number of samples used in the training set
            2. 2D numpy.array of size N_SAMPLES x N_CAT_PARAMS where N_CAT_PARAMS is the number of nominal/ordinal
                parameters and N_SAMPLES is the number of samples used in the training set
            3. 1D numpy.array of size N_SAMPLES containing the target values used for training

        cv_set: tuple of 3 elements
            Similar to train_set but this time N_SAMPLES is the number of samples in the cv set

        test_set: tuple of 3 elements
            Similar to train_set but this time N_SAMPLES is the number of samples in the cv set
        """

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
        """
        - Data normalization if discrete/continuous parameters + target
        - One-hot-encoder if categorical parameters
        - Merge discrete/continuous and categorical parameters

        Returns
        ----------------
        X: tuple of 3 elements
            There is one element for each train, cv and test set. Each element corresponds to a 2D numpy.array of size
            N_SAMPLES x N_PARAMETERS where N_SAMPLES changes in function of the set but N_PARAMETERS is always the same
            N_PARAMETERS = N_CONT_PARAMETERS + N_CAT_PARAMETERS (after one-hot-encoding)

        y: tuple of 3 elements
            Each element (1D numpy.array) corresponds to the target values for training, cv and testing set.

        mean_sale_price: float
            Mean value used to do the z-normalization of the target values

        std_sale_price: float
            Standard deviation value used to do the z-normalization of the target values
        """

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

        return (X_train, X_cv, X_test), (self.target_train, self.target_cv, self.target_test), mean_sale_price, std_sale_price