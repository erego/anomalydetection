import unittest

import pandas as pd
import scipy.io

from anomalydetection.data.calculations import pca_analysis
from anomalydetection.data.processing import feature_normalization


class TestPCAAlgorithm(unittest.TestCase):

    # Test parameter multivariate gaussian
    @staticmethod
    def test_fit_parameter_multivariate_gaussian():

        # Read data from file
        mat_data = scipy.io.loadmat('data/ex7data1.mat')
        x_data = mat_data['X']

        x_norm = feature_normalization(x_data, 'minmax')
        pca_model = pca_analysis(x_norm)
        assert pca_model.explained_variance_[0] == 0.09866204761884817
        assert pca_model.explained_variance_[1] == 0.014895047190436293

        x_norm = pd.DataFrame(pca_model.transform(x_norm))

        first_column = x_norm[0]
        second_column = x_norm[1]

        assert first_column[0] == 0.3355334492544746
        assert second_column[0] == -0.18275596898151228
