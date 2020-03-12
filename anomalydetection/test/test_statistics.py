from unittest import TestCase

import pandas as pd
import scipy.stats as sts


class StatisticsTest(TestCase):

    df_data = pd.read_csv('./data/Iris.csv', sep=',', header=0, skiprows=0)

    def test_levene_student(self):

        data = self.df_data['SepalLengthCm']
        data_first = data[0:100]
        data_second = data[120:140]

        # Test de Levene
        (st_levene, p_levene) = sts.levene(data_first, data_second)
        self.assertEqual(st_levene, 0.010954529419409475)
        self.assertEqual(p_levene, 0.9168202144934579)

        # Test de Student

        # Same variance
        if p_levene > 0.05:
            (st_student, p_student) = sts.ttest_ind(data_first, data_second, equal_var=True)
        # Not same variance
        else:
            (st_student, p_student) = sts.ttest_ind(data_first, data_second, equal_var=False)

        self.assertEqual(st_student, -7.71539089713345)
        self.assertEqual(p_student, 4.264324342451045e-12)

        reject_h0 = False
        if p_student < 0.05:
            reject_h0 = True

        self.assertTrue(reject_h0)

    def test_chi_square(self):
        row1 = [91, 90, 51]
        row2 = [150, 200, 155]
        row3 = [109, 198, 172]
        data = [row1, row2, row3]
        chi_square_result = sts.chi2_contingency(data)
        self.assertEqual(chi_square_result[2], 4)
        self.assertEqual(chi_square_result[1], 4.8346447416999636e-05)

