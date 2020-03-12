import math

import numpy as np

from anomalydetection.models.model import Model


class IndependentUnivariateGaussian(Model):

    # Initialise data
    def __init__(self, data_dictionary):

        if 'cross_validation' not in data_dictionary:
            raise ValueError("No cross validation data in dictionary. A cross validation data "
                             "is necessary")

        if 'cross_validation_output' not in data_dictionary:
            raise ValueError("No cross validation output datta in dictionary. A cross validation "
                             "output data is necessary")

        super(IndependentUnivariateGaussian, self).__init__(data_dictionary)
        self.mean = 0
        self.std = 0

    def fit_parameter(self):

        # Check if propertyId column is in data, in that case extract the column
        data_to_fit = self.df_train
        if 'propertyId' in self.df_train.columns:
            data_to_fit = data_to_fit.drop('propertyId', 1)

        (self.mean, self.std) = IndependentUnivariateGaussian.fit_parameter_model(
            data_to_fit.values)

    def get_probabilities(self, data):

        # Check if propertyId column is in data, in that case extract the column
        data_to_get = data
        if 'propertyId' in data_to_get.columns:
            data_to_get = data_to_get.drop('propertyId', 1)

        p_validation = IndependentUnivariateGaussian.get_probabilities_iuvg(
            data_to_get.values, self.mean, self.std)
        return p_validation

    # Fit parameters
    @staticmethod
    def fit_parameter_model(df):

        mean = np.mean(df, axis=0)
        std = np.std(df, axis=0)

        return mean, std

    # Compute normal gaussian probability for a sample
    @staticmethod
    def compute_probability_iuvg(sample, mean, std):

        sample_dif = np.subtract(sample, mean)
        sample_dif_sqr = np.multiply(sample_dif, sample_dif)
        variance = np.multiply(std, std)

        exp_term = -np.divide(sample_dif_sqr, 2 * variance)
        exp = np.exp(exp_term)

        exp_divide_sigma = np.divide(exp, std)

        # print exp_divide_sigma

        final_value = (1 / math.sqrt(2.0 * math.pi)) * exp_divide_sigma

        # print final_value

        probabilities = np.prod(final_value)
        # final_value =  np.log(final_value)
        # probabilities = np.sum(final_value)

        return probabilities

    # Return a vector of probabilities of each sample
    @staticmethod
    def get_probabilities_iuvg(data, mean, std):

        probabilities_vector = []

        for row in data:
            pb = IndependentUnivariateGaussian.compute_probability_iuvg(row, mean, std)
            probabilities_vector.append(pb)

        return probabilities_vector
