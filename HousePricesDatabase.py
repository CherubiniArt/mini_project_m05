import pandas as pd
import numpy as np
import random


class HousePricesDatabase():

    def __init__(self, database_path, continuous_parameters, discrete_parameters, ordinal_parameters,
                 nominal_parameters, protocol=[0.6, 0.2, 0.2]):
        """
        Attributes
        -----------
        database_path: str
            complete path to the database file

        continuous_parameters: list of str
            list containing the name of all the continuous parameters taken into account for this experiment

        discrete_parameters: list of str
            list containing the name of all the discrete parameters taken into account for this experiment

        ordinal_parameters: list of str
            list containing the name of all the ordinal parameters taken into account for the experiment

        nominal_parameters: list of str
            list containing the name of all the nominal parameters taken into account for the experiment

        protocol: list of float
            a list of coefficients defining relative sizes of training, cv, and test sets
            Default: [0.6, 0.2, 0.2]
        """

        self.database_path = database_path
        self.continuous_parameters = continuous_parameters
        self.discrete_parameters = discrete_parameters
        self.ordinal_parameters = ordinal_parameters
        self.nominal_parameters = nominal_parameters
        self.protocol = protocol

        pd.options.mode.chained_assignment = None

    def __call__(self):
        """
        Read the data from a csv fil
        Clean the data (cf NaN value)
        Split the dataset into training, cv and testing sets

        Returns
        -----------------------
        train_set: tuple of 3 elements
            1. 2D numpy.array of size N_SAMPLES x N_CONT_PARAMS where N_CONT_PARAMS is the number of discrete/continuous
                parameters and N_SAMPLES is the number of samples used in the training set
            2. 2D numpy.array of size N_SAMPLES x N_CAT_PARAMS where N_CAT_PARAMS is the number of nominal/ordinal
                parameters and N_SAMPLES is the number of samples used in the training set
            3. 1D numpy.array of size N_SAMPLES containing the target values used for training

        cv_set: tuple of 2 elements
            Similar to train_set but this time N_SAMPLES is the number of samples in the cv set

        test_set: tuple of 2 elements
            Similar to train_set but this time N_SAMPLES is the number of samples in the test set
        """

        dataset = pd.read_csv(self.database_path)

        # Split the dataset into two parts: one containing continuous/discrete params and one containing categorical
        # params
        continuous_dataset = dataset[self.continuous_parameters + self.discrete_parameters].to_numpy()
        categorical_dataset = dataset[self.ordinal_parameters + self.nominal_parameters]
        target = dataset["SalePrice"].to_numpy()

        # First the NaN value in the categorical dataset are replaced by None:
        categorical_dataset.fillna('None', inplace=True)
        categorical_dataset = categorical_dataset.to_numpy()

        # Get the nan index in the continuous dataset
        nan_index = np.argwhere(np.isnan(continuous_dataset))[:, 0]

        # Delete the line where there is a nan_index:
        continuous_dataset = np.delete(continuous_dataset, nan_index, axis=0)
        categorical_dataset = np.delete(categorical_dataset, nan_index, axis=0)
        target = np.delete(target, nan_index, axis=0)

        #=======================================================================================
        # Split the dataset into train, cv and test
        # The samples are randomly, but with seeding, shuffled before splitting
        n_sample = target.shape[0]

        n_sample_train = int(n_sample*self.protocol[0])
        n_sample_cv = int(n_sample * self.protocol[1])

        index_list = list(np.arange(n_sample))
        random.Random(7).shuffle(index_list)

        train_index = sorted(index_list[:n_sample_train])
        cv_index = sorted(index_list[n_sample_train:n_sample_train+n_sample_cv])
        test_index = sorted(index_list[n_sample_train+n_sample_cv:])

        # =======================================================================================
        train_set = (continuous_dataset[train_index, :], categorical_dataset[train_index, :], target[train_index])
        cv_set = (continuous_dataset[cv_index, :], categorical_dataset[cv_index, :], target[cv_index])
        test_set = (continuous_dataset[test_index, :], categorical_dataset[test_index, :], target[test_index])

        return train_set, cv_set, test_set
