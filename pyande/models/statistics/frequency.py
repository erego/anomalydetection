import numpy as np

from pyande.models.model import Model
from pyande.data.calculations import get_optimus_bins


class Frequency(Model):

    # Initialise data
    def __init__(self, data_dictionary):
        super(Frequency, self).__init__(data_dictionary)

        self.model = None

    # Fit parameter according to training set
    def fit_parameter(self):

        self.model = Frequency.fit_parameter_model(self.df_train)

    def get_probabilities(self, data):

        data_to_get_probabilities = data

        p_validation = Frequency.get_probabilities_model(data_to_get_probabilities,
                                                         self.model, len(self.df_train))

        return p_validation

    @staticmethod
    # Fit parameters
    def fit_parameter_model(df):
        df_values = df.values

        m = len(df_values[0, :])
        rows_number = len(df.index)
        frequency_model = dict()

        for i in range(m):
            feature_value = df_values[:, i]

            num_bins = get_optimus_bins(feature_value, m)

            hist, bin_edges = np.histogram(feature_value, num_bins)

            frequencies = hist / rows_number
            frequency_model[df.columns[i]] = (frequencies, bin_edges)

        return frequency_model

    @staticmethod
    def get_probabilities_model(sample, frequency_model, train_length):
        probabilities_vector = []

        # Go through each row and compute probability by row
        for row in sample.values:
            pb = 1
            it_row = np.nditer(row, flags=['multi_index'])
            while not it_row.finished:
                column_name = sample.columns[it_row.multi_index[0]]
                column_value = it_row[0]

                # Find bin that match to this value for column "column_name"
                (frequency_data, bin_data) = frequency_model[column_name]

                # What bin best fit to data?
                part_prob = 0

                length_fd = len(frequency_data)
                for i in range(length_fd):
                    left_value = bin_data[i]
                    right_value = bin_data[i + 1]
                    if i == length_fd - 1:
                        if (left_value <= column_value) and (right_value >= column_value):
                            part_prob = frequency_data[i]
                    else:
                        if (left_value <= column_value) and (right_value > column_value):
                            part_prob = frequency_data[i]
                            break

                if part_prob == 0:
                    # sub = frequency_data[np.where(frequency_data > 0)]
                    # part_prob = np.min(sub)

                    part_prob = 1.0 / train_length

                pb = pb * part_prob

                it_row.iternext()

            probabilities_vector.append(pb)

        return probabilities_vector
