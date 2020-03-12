import math
import numpy as np

from pyande.models.model import Model
from pyande.data.draw import draw_correlation_matrix


class MultivariateGaussian(Model):

    # Initialise data
    def __init__(self, data_dictionary):

        if 'cross_validation' not in data_dictionary:
            raise ValueError("No cross validation data in dictionary. A cross validation data "
                             "is necessary")

        if 'cross_validation_output' not in data_dictionary:
            raise ValueError("No cross validation output datta in dictionary. A cross validation "
                             "output data is necessary")

        super(MultivariateGaussian, self).__init__(data_dictionary)

        self.mean = 0
        self.sigma = 0

    def fit_parameter(self):

        (self.mean, self.sigma) = MultivariateGaussian.fit_parameter_model(self.df_train)

    def get_probabilities(self, data):

        p_validation = MultivariateGaussian.get_probabilities_mvg(data.values, self.mean,
                                                                  self.sigma)
        return p_validation

    @staticmethod
    def fit_parameter_model(data):

        mean = np.mean(data, axis=0)
        sigma = np.cov(data, rowvar=False)

        return mean, sigma

    @staticmethod
    def compute_probability_mvg(sample, mean, sigma, det_sigma):

        if det_sigma == 0:
            raise NameError("The covariance matrix can't be singular")

        const_divide = math.pow((2.0 * np.pi), (len(sample) / 2.0)) * math.pow(det_sigma, 1.0 / 2)
        const_divide = 1.0 / const_divide

        sample_dif = np.subtract(sample, mean)

        if sigma.size == 1:
            sigma_inverse = 1.0 / sigma
        else:
            sigma_inverse = np.linalg.inv(sigma)
        prod_matrix = sample_dif.dot(sigma_inverse)
        prod_matrix = prod_matrix.dot(sample_dif.T)

        result = math.exp(-0.5 * prod_matrix)

        probabilities = const_divide * result

        return probabilities

    @staticmethod
    def get_probabilities_mvg(sample, mean, sigma):
        probabilities_vector = []

        # Determinant calculations
        if sigma.size == 1:
            det_sigma = sigma
        else:

            det_sigma = np.linalg.det(sigma)
            if det_sigma < 0:
                raise NameError("The covariance matrix is negative. Try to normalise features or "
                                "review features values.")

        for row in sample:
            pb = MultivariateGaussian.compute_probability_mvg(row, mean, sigma, det_sigma)
            probabilities_vector.append(pb)

        return probabilities_vector

    def draw_correlation_matrix(self, data):
        draw_correlation_matrix(self.sigma, data)
