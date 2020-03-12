import numpy as np
from sklearn import preprocessing
import pandas as pd


def feature_categorical_conversion(lst_data):

    """
    Convert categorical data in numerical from a list of dataset

    :param lst_data: list of related dataset to convert in numerical
    :return: list of converted data sets and dictionary of conversion categorical in numerical
    """

    dict_categorical_features = dict()

    features_all = lst_data[0].columns.values.tolist()

    for feature in features_all:
        dict_values = dict()
        lst_values = []

        for data in lst_data:
            df_data_feature = data[feature]

            if df_data_feature.dtype == object:

                # To convert into numbers
                # Create a dictionary between factors an numbers
                data[feature].replace(np.NaN, 'Unknown', inplace=True)

                # Get unique values and added to list of values if not exists and in dictionary
                lst_values_data = data[feature].unique()
                lst_values_added = [item for item in lst_values_data if item not in lst_values]

                lst_values.extend(lst_values_added)

                for value in lst_values_added:
                    dict_values[value] = len(dict_values)

                # Convert categorical data to number
                data[feature].replace(dict_values, inplace=True)

        if len(lst_values) > 0:
            dict_values = dict()
            for value in lst_values:
                dict_values[value] = len(dict_values)

            dict_categorical_features[feature] = dict_values

    return lst_data, dict_categorical_features


def feature_combination(df_train, df_cross_validation,  df_normal, df_anomaly):

    new_data_train = pd.DataFrame()
    new_data_cross_validation = pd.DataFrame()
    new_data_normal = pd.DataFrame()
    new_data_anomaly = pd.DataFrame()

    for i in range(len(df_train.columns)):
        for j in range(i+1, len(df_train.columns)):
            column_name = "C(" + str(i) + "," + str(j) + ")"
            data_train_product = df_train.ix[:, i] * df_train.ix[:, j]
            data_cross_validation_product = \
                df_cross_validation.ix[:, i] * df_cross_validation.ix[:, j]
            if len(pd.Series.unique(data_train_product)) != 1 and \
                    len(pd.Series.unique(data_cross_validation_product)) != 1:

                new_data_train[column_name] = df_train.ix[:, i] * df_train.ix[:, j]
                new_data_cross_validation[column_name] = \
                    df_cross_validation.ix[:, i] * df_cross_validation.ix[:, j]
                new_data_normal[column_name] = df_normal.ix[:, i] * df_normal.ix[:, j]
                new_data_anomaly[column_name] = df_anomaly.ix[:, i] * df_anomaly.ix[:, j]

    data_train_result = pd.concat([df_train, new_data_train], axis=1)
    data_cross_validation_result = pd.concat([df_cross_validation, new_data_cross_validation],
                                             axis=1)
    data_normal_result = pd.concat([df_normal, new_data_normal], axis=1)
    data_anomaly_result = pd.concat([df_anomaly, new_data_anomaly], axis=1)

    return data_train_result, data_cross_validation_result, data_normal_result, data_anomaly_result


def feature_normalization(data, type_normalization):
    """
    Normalise range data according selected type (std, minmax, average mean)
    :param data: data to normalise
    :param type_normalization: type of normalization
    :return:
    """

    if type_normalization == "std":
        std_scale = preprocessing.StandardScaler().fit(data)
        data_norm = std_scale.transform(data)
    elif type_normalization == "minmax":
        std_scale = preprocessing.MinMaxScaler().fit(data)
        data_norm = std_scale.transform(data)
    else:
        mean = np.mean(data, axis=0)

        data_norm = data - mean

        std = np.std(data_norm, axis=0)

        data_norm = data_norm / std

    return data_norm


def set_gaussian_shape(df_data, df_feature, factor_to_multiply, funct_to_apply, *args):
    """
    This function try to make feautre distribution loos like a gaussian shape to use Bayes theorem
    :param df_data: data frame with data
    :param df_feature: feature of data frame to modify
    :param factor_to_multiply: factor to modify if necessary
    :param funct_to_apply:function to apply in order to modify original shape of
    feature(log, power,...)
    :param args: arguments of function to apply if is necessary
    :return: data frame with value from feature selected modified
    """

    df_data[df_feature] = factor_to_multiply * funct_to_apply(df_data[df_feature], args)
