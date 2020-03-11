from unittest import TestCase
from math import pi, cos, sin
from random import random

import pandas as pd

from anomalydetection.models.svm import OutlierSVM


def calculate_point_circle():
    theta = random() * 2 * pi
    r = random()
    return cos(theta) * r, sin(theta) * r


class TestSVM(TestCase):

    def test_svm(self):
        points_in_circle = [calculate_point_circle() for _ in range(100)]

        df_normal = pd.DataFrame(points_in_circle, columns=['X', 'Y'])

        anomaly_points = [(4, 4), [-7, -7]]

        df_anomaly = pd.DataFrame(anomaly_points, columns=['X', 'Y'])

        total_points = points_in_circle + anomaly_points

        df_train = pd.DataFrame(total_points, columns=['X', 'Y'])

        data_dictionary = {'train': df_train, 'normal': df_normal, 'anomaly': df_anomaly,
                           'nu': 0.05}

        svm_model = OutlierSVM(data_dictionary)
        svm_model.fit_parameter()

        (f1_score, accuracy, tp, fp, fn, tn) = svm_model.get_predictions()
        self.assertEqual(f1_score, 0.038461538461538464)
        self.assertEqual(tp, 2)

