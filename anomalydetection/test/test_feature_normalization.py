from unittest import TestCase

import numpy as np

from anomalydetection.data import processing


class TestFeatureNormalization(TestCase):

    # Test Feature Normalization
    def test_feature_normalization(self):

        data = np.array([[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]])
        data_norm = processing.feature_normalization(data, "std")
        data_result = np.array([[0., -1.224,  1.336], [1.224,  0., -0.267],
                                [-1.224,  1.224, -1.069]])
        np.testing.assert_array_almost_equal(data_norm, data_result, decimal=2)
