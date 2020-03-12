import collections

import numpy as np

from pyande.models.model import Model
from pyande.models.statistics import mvg


class MultivariateGaussianRelativeFrequency (Model):
    # Initialise data
    def __init__(self, data_dictionary):
        super(MultivariateGaussianRelativeFrequency, self).__init__(data_dictionary)

        self.mean = 0
        self.sigma = 0
        self.frequency = None

    # Fit parameter according to training set
    def fit_parameter(self):

        # Get subset from data set with continuous features
        df_train_continuous = self.df_train[self.CONTINUOUS_FEATURE]
        (self.mean, self.sigma) = mvg.MultivariateGaussian.fit_parameter_model(
            df_train_continuous.values)

        # Get subset from data set with discrete features
        df_train_discrete = self.df_train[self.DISCRETE_FEATURE]
        self.frequency = MultivariateGaussianRelativeFrequency.fit_parameter_relative_frequency(
            df_train_discrete.values)

    def get_probabilities(self, data):

        data_continuous = data[self.CONTINUOUS_FEATURE]

        p_validation_mvg = mvg.MultivariateGaussian.get_probabilities_mvg(data_continuous.values,
                                                                          self.mean, self.sigma)
        data_discrete = data[self.DISCRETE_FEATURE]

        p_validation_rf = MultivariateGaussianRelativeFrequency.\
            get_probabilities_relative_frequency(data_discrete.values, self.frequency,
                                                 len(self.df_train))

        p_validation = [p_validation_mvg[i] * p_validation_rf[i] for i in
                        range(len(p_validation_mvg))]

        return p_validation

    @staticmethod
    def fit_parameter_relative_frequency(data):

        num_samples = len(data)

        prob_columns = []

        for column in data.T:
            c = collections.Counter(column)
            for key, value in c.items():
                c[key] = value / num_samples
            prob_columns.append(c)

        return prob_columns

    @staticmethod
    def compute_probability_relative_frequency(sample):

        probabilities = np.prod(sample)

        return probabilities

    @staticmethod
    def get_probabilities_relative_frequency(sample, frequency, train_length):

        probabilities_vector = []

        data_prob = np.empty(sample.shape, dtype=float)

        it = np.nditer(sample, flags=['multi_index'])
        while not it.finished:

            # find frequency in list freq_columns position it.multi_index[1]
            prob_item = frequency[it.multi_index[1]]

            cell_value = sample[it.multi_index[0], it.multi_index[1]]

            if cell_value in prob_item:
                prob_value = prob_item[cell_value]
            else:
                # prob_value = 0
                prob_value = 1.0 / train_length

            data_prob[it.multi_index[0], it.multi_index[1]] = prob_value

            it.iternext()

        for row in data_prob:
            pb = MultivariateGaussianRelativeFrequency.compute_probability_relative_frequency(row)
            probabilities_vector.append(pb)

        return probabilities_vector
