from sklearn import svm

from anomalydetection.models.model import Model


class OutlierSVM(Model):
    """
    Class to detect outlier with SVM
    Train data only contains normal data
    """

    # Initialise data
    def __init__(self, data_dictionary):

        if 'normal' not in data_dictionary:
            raise ValueError("No normal data in dictionary. A normal data is necessary")

        if 'anomaly' not in data_dictionary:
            raise ValueError("No anomaly data in dictionary. A anomaly data is necessary")

        nu_parameter = 0.95*0.15+0.05
        if 'nu' in data_dictionary:
            nu_parameter = data_dictionary['nu']

        super(OutlierSVM, self).__init__(data_dictionary)

        self.model = svm.OneClassSVM(nu=nu_parameter, kernel="rbf", gamma=0.1)

    # Fit parameter according to training set
    def fit_parameter(self):
        df_train_n = self.df_train
        self.model.fit(df_train_n)

    def get_predictions(self):
        """
        Get prediction measures from data
        :return: f1 score, true positive, false positive, false negative, true negative
        """

        df_normal_n = self.df_normal
        df_anomaly_n = self.df_anomaly

        y_pred_normal = self.model.predict(df_normal_n)
        y_pred_anomaly = self.model.predict(df_anomaly_n)

        n_outliers_normal = y_pred_normal[y_pred_normal == -1].size
        n_outliers_anomaly = y_pred_anomaly[y_pred_anomaly == -1].size

        # True positive (its anomalous and predict as anomalous)
        tp = n_outliers_anomaly

        # False positive (its normal and predict as anomalous)
        fp = n_outliers_normal

        # False negative (its anomalous and predict as normal)
        fn = y_pred_anomaly.size - n_outliers_anomaly

        # True negative (its normal and predict as normal)
        tn = y_pred_normal.size - n_outliers_normal

        f1_score = (2*tp)/(2*tp + fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return f1_score, accuracy, tp, fp, fn, tn
