"""
Class of different utilities related to calculations useful for models
"""

import numpy as np


def get_optimus_bins(feature_values, number_of_features):
    """
    Get the optimus number of bins for a set of values
    :param feature_values: list of feature values
    :param number_of_features: number of features in the dataset
    :return: number of bins optimus
    """

    max_value = max(feature_values)
    min_value = min(feature_values)

    num_of_different_values = len(np.unique(feature_values))

    if num_of_different_values <= 10:
        num_bins = len(np.unique(feature_values))
    else:
        # Freedman- Diaconis num of bins
        q75, q25 = np.percentile(feature_values, [75, 25])
        iqr = q75 - q25

        if iqr < (max_value - min_value) / 10:
            iqr = (max_value - min_value) / 10

        h_bins = 2 * iqr * np.power(number_of_features, -1 / 3)

        num_bins = (max_value - min_value) / h_bins
        num_bins = int(np.ceil(num_bins))

    return num_bins

# PCA Analysis
def pca_analysis(data):

    pca_model = pca_analysis_scikit(data)
    return pca_model


def pca_analysis_scikit(data):

    pca = PCA()
    pca.fit(data)
    list_variance = pca.explained_variance_ratio_.tolist()

    sum_variance = 0
    num_sum = 0
    for number in list_variance:
        sum_variance = sum_variance + number
        num_sum = num_sum + 1
        if sum_variance > 0.95:
            break

    ncomponents = num_sum
    pca = PCA(n_components=ncomponents)
    pca.fit(data)

    return pca