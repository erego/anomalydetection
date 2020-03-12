from unittest import TestCase

from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.io

from pyande.models.statistics.iuvg import IndependentUnivariateGaussian
from pyande.data.calculations import select_threshold


class TestIUVG(TestCase):

    def test_mvg(self):

        mat = scipy.io.loadmat('./data/cardio.mat')
        x_data = mat['X']
        y_data = mat['y']

        header_list = ["LB-FHR", "AC", "FM", "UC", "DL", "DS", "ASTV", "MSTV", "ALTV", "MLTV",
                       "width", "min", "max", "nmax", "nzeros", "mode", "mean", "median",
                       "variance", "tendency", "class-fhr"]

        data = pd.DataFrame(x_data, columns=header_list)
        data = data.iloc[:, 0:4]

        x_train, x_test, y_train, y_test = train_test_split(data, y_data, test_size=0.2,
                                                            random_state=1)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                          random_state=1)

        data_dictionary = {'train': x_train, 'cross_validation': x_val,
                           'cross_validation_output': y_val, 'test': x_test}

        statistics_model = IndependentUnivariateGaussian(data_dictionary)
        statistics_model.fit_parameter()

        self.assertEqual(statistics_model.mean[0], -0.009726763045597591)

        self.assertEqual(statistics_model.std[0], 1.0055511594794753)

        cross_validation_values = statistics_model.get_probabilities(x_val)

        (epsilon, f1, complexity) = select_threshold(y_val.flatten(), cross_validation_values)

        self.assertEqual(epsilon, 3.468655474407142e+21)
        self.assertEqual(f1, 0.2620689655172414)
