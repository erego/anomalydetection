from unittest import TestCase

from anomalydetection.data.calculations import get_optimus_bins


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
