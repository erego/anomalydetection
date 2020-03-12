from unittest import TestCase

import numpy as np
import scipy.io

from pyande.data.calculations import get_optimus_bins, select_threshold, \
    get_statistical_measures
from pyande.models.statistics.mvg import MultivariateGaussian


class TestCalculations(TestCase):

    def test_bin_normal(self):

        m = 7
        test_values = [1] * 10 + [2] * 5

        num_bins = get_optimus_bins(test_values, m)

        self.assertEqual(num_bins, 2)

    def test_bin_freedman_diaconis(self):

        m = 22
        test_values = [x for x in range(20)] * 2

        num_bins = get_optimus_bins(test_values, m)

        self.assertEqual(num_bins, 3)

    def test_threshold(self):
        y_val = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        p_val = [1.1394960147909029e-10, 6.358713373831626e-10, 1.8974411442399827e-15,
                 1.7798775956176128e-31, 2.954521012223191e-11, 9.406973485322113e-13,
                 3.128266755817403e-13, 7.959266011854885e-12, 2.627310407677617e-10,
                 1.1860958371180245e-11, 2.137439716790242e-12, 5.159084274239081e-13,
                 9.92681851599294e-08]

        (epsilon, f1, complexity) = select_threshold(y_val, p_val)
        self.assertEqual(epsilon, 1.7916203624944257e+21)
        self.assertEqual(f1, 0.8571428571428571)

    def test_statistical_measures_data(self):
        # Read data from file
        mat_data = scipy.io.loadmat('data/ex8data1.mat')

        x_data = mat_data['X']

        x_val = mat_data['Xval']
        y_val = mat_data['yval']

        (mean, std) = MultivariateGaussian.fit_parameter_model(x_data)

        p = MultivariateGaussian.get_probabilities_mvg(x_data, mean, std)

        p_val = MultivariateGaussian.get_probabilities_mvg(x_val, mean, std)

        total_positives, total_negatives, statistical_measures = get_statistical_measures(
            y_val[:, 0], p_val)

        [epsilon, f1, _] = select_threshold(y_val[:, 0], p_val)

        self.assertEqual(total_positives, 9)
        self.assertEqual(total_negatives, 298)

        self.assertEqual(epsilon, 1.5694315600331694e+35)
        self.assertEqual(f1, 0.8750000000000001)
