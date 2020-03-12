"""
Class of different utilities related to calculations useful for models
"""

import math

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


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


def select_threshold(y_values, p_values):
    """
        select_threshold Find the best threshold (epsilon) to use for selecting outliers
        [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
        threshold to use for selecting outliers based on the results from a
        validation set (pval) and the ground truth (yval).
    """

    best_epsilon = 0
    best_f1 = 0
    best_tp = 0
    best_tn = 0
    best_fp = 0
    best_fn = 0

    probability_list = pd.DataFrame(
        {'pval': p_values,
         'yval': y_values
         })

    pval = probability_list['pval'].tolist()
    yval = probability_list['yval'].values

    probability_list = probability_list[probability_list.pval != 0]

    val_to_sum = min(probability_list['pval'].tolist())

    # Normalize pval
    pval = [y + val_to_sum for y in pval]
    pval = [math.log(y, 10) for y in pval]

    min_pval = min(pval)

    pval = [y - min_pval for y in pval]
    min_pval = min(pval)
    max_pval = max(pval)
    stepsize = (max_pval - min_pval) / 10000

    rng = np.arange(min_pval, max_pval, stepsize)

    for epsilon in rng:

        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note: You can use predictions = (pval < epsilon) to get a binary vector
        #       of 0's and 1's of the outlier predictions

        pval = np.array(pval)
        predictions = (pval < epsilon)

        tp = sum((predictions == 1) & (yval == 1))

        tn = sum((predictions == 0) & (yval == 0))

        fp = sum((predictions == 1) & (yval == 0))

        fn = sum((predictions == 0) & (yval == 1))

        precision = tp/(tp + fp)
        recall = tp/(tp + fn)

        sensitivity = tp/(tp + fn)
        specificity = tn/(tp + fn)

        f1 = (2*precision*recall)/(precision + recall)

        if f1 > best_f1:
            precision_f1 = precision
            recall_f1 = recall
            best_f1 = f1
            best_epsilon = epsilon
            best_tp = tp
            best_tn = tn
            best_fp = fp
            best_fn = fn
            best_predictions = predictions

    print("confusion matrix")

    print(confusion_matrix(yval, best_predictions, labels=[1, 0]))

    print("tp, tn, fp, fn")
    print(best_tp, best_tn, best_fp, best_fn)

    lst_complexity = []

    pval_order = sorted(pval)

    if len(pval_order) >= 100:
        num_elements = int((1 * len(pval_order))/100)
    else:
        num_elements = int((1 * len(pval_order)) / 10)

    first_elements = pval_order[:num_elements]
    last_elements = pval_order[len(pval_order) - num_elements:]
    lower = max(first_elements)
    upper = min(last_elements)
    # (lower,upper) = st.t.interval(0.97, len(pval)-1, loc=np.mean(pval), scale=st.sem(pval))

    lst_complexity.append(lower)

    incr = (upper - lower) / 5.0

    lst_complexity.append(lower + incr)
    lst_complexity.append(lower + 2 * incr)
    lst_complexity.append(lower + 3 * incr)
    lst_complexity.append(upper - 2 * incr)
    lst_complexity.append(upper - incr)
    lst_complexity.append(upper)

    lst_complexity = [x + min_pval for x in lst_complexity]
    lst_complexity = [math.pow(10, x) for x in lst_complexity]
    lst_complexity = [x - val_to_sum for x in lst_complexity]

    best_epsilon = best_epsilon + min_pval
    best_epsilon = math.pow(10, best_epsilon)
    best_epsilon = best_epsilon - val_to_sum
    print("best_epsilon")
    print(best_epsilon)

    return best_epsilon, best_f1, lst_complexity
