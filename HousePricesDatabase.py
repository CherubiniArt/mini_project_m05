import pandas as pd
import numpy as np
import random


class HousePricesDatabase():

    def __init__(self, database_path, continuous_parameters, discrete_parameters, ordinal_parameters,
                 nominal_parameters, protocol=[0.6, 0.2, 0.2]):

        self.database_path = database_path
        self.continuous_parameters = continuous_parameters
        self.discrete_parameters = discrete_parameters
        self.ordinal_parameters = ordinal_parameters
        self.nominal_parameters = nominal_parameters
        self.protocol = protocol

    def __call__(self):
        """
        Return: train, cv, test

        train, cv, test are dict where the keys are data and meta-data
        """

        dataset = pd.read_csv(self.database_path)

        # Split the dataset into two parts: one containing continuous/discrete params and one containing categorical
        # params
        continuous_dataset = dataset[self.continuous_parameters + self.discrete_parameters].to_numpy()
        categorical_dataset = dataset[self.ordinal_parameters + self.nominal_parameters]
        target = dataset["SalePrice"].to_numpy()

        # First the NaN value in the categorical dataset are replaced by None:
        categorical_dataset.fillna('None', inplace=True)

        # The different str categories have to be transformed into integer:
        """for param in self.ordinal_parameters + self.nominal_parameters:
            # Get all the str categories for each param
            categories = np.unique(np.asarray(categorical_dataset[param]))

            # Replace a str categories by a integer value
            for i, cat in enumerate(categories):
                categorical_dataset[param].replace(to_replace=cat, value=i, inplace=True)"""

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