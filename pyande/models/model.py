import numpy as np


class Model(object):

    # Initialise data
    def __init__(self, data_dictionary):
        """
        Class to represent a model
        :param data_dictionary: Dictionary with all data necessary for the model
        """

        self.CONTINUOUS_FEATURE = []
        self.DISCRETE_FEATURE = []
        self.dict_categorical_features = dict()

        if 'train' not in data_dictionary:
            raise ValueError("No train data in dictionary. A train data is necessary")

        self.df_train = data_dictionary["train"]

        self.df_cross_validation = None
        if "cross_validation" in data_dictionary:
            self.df_cross_validation = data_dictionary["cross_validation"]

        self.df_cross_validation_output = None
        if "cross_validation_output" in data_dictionary:
            self.df_cross_validation_output = data_dictionary["cross_validation_output"]

        self.df_normal = None
        if "normal" in data_dictionary:
            self.df_normal = data_dictionary["normal"]

        self.df_anomaly = None
        if "anomaly" in data_dictionary:
            self.df_anomaly = data_dictionary["anomaly"]

        self.get_features_type_list()

    # Get Continuous and Discrete Features List for the data
    def get_features_type_list(self):

        features_all = self.df_train.columns.values.tolist()

        rows_number = len(self.df_train)

        for feature in features_all:
            df_data_feature = self.df_train[feature]

            values_number = len(np.unique(df_data_feature))

            # If the number of unique feature values for that column is less than 20% of total rows
            # feature is considered discrete, otherwise continue
            if values_number < 0.2 * rows_number:
                self.DISCRETE_FEATURE.append(feature)
            else:
                self.CONTINUOUS_FEATURE.append(feature)

    # TODO why here?
    # Return a vector of complexity of each sample
    @staticmethod
    def get_complexity(probabilities, complexity):

        complexity_vector = []

        for prob in probabilities:

            if prob < complexity[0]:
                comp = 6
            elif prob < complexity[1]:
                comp = 5
            elif prob < complexity[2]:
                comp = 4
            elif prob < complexity[3]:
                comp = 3
            elif prob < complexity[4]:
                comp = 2
            elif prob < complexity[5]:
                comp = 1
            else:
                comp = 0
            complexity_vector.append(comp)

        return complexity_vector

    # TODO Why here?
    # Return a vector of anomalous of each sample
    @staticmethod
    def get_anomalous(probabilities, epsilon):
        anomalous_vector = []

        for prob in probabilities:

            if prob < epsilon:
                anomalous = 'Y'
            else:
                anomalous = 'N'

            anomalous_vector.append(anomalous)

        return anomalous_vector

    # Set dict of categorical transformation
    def set_categorical_features_transformation(self, dict_transformation):

        self.dict_categorical_features = dict_transformation
