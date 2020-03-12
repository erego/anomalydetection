import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mptches
import seaborn as sns
import scipy.stats as sts


def draw_correlation_matrix(sigma, data):

    # Get correlation matrix and draw it
    corr_matrix = np.corrcoef(sigma)
    features = data.columns.values.tolist()

    sns.set(style="white")
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=features, yticklabels=features)

    plt.xlabel('Correlation Matrix')
    plt.show()


def draw_roc_curve(total_positives, total_negatives, statistical_measures, cost_fp, cost_fn):

    total_population = total_positives + total_negatives

    # epsilon
    epsilon = statistical_measures[:, 0]

    # RATIO
    # True positive ratio
    tp_ratio = statistical_measures[:, 1] / total_population

    # False positive ratio
    fp_ratio = statistical_measures[:, 2] / total_population

    # False negative ratio
    fn_ratio = statistical_measures[:, 3] / total_population

    # True negative ratio
    tn_ratio = statistical_measures[:, 4] / total_population

    # RATE
    # True positive rate
    tp_rate = statistical_measures[:, 1] / total_positives

    # False positive rate
    fp_rate = statistical_measures[:, 2] / total_negatives

    cost = fp_ratio * cost_fp + fn_ratio * cost_fn

    plt.figure()

    red_patch = mptches.Patch(color='red', label='The false positive ratio')
    blue_patch = mptches.Patch(color='blue', label='The false negative ratio')
    plt.legend(handles=[red_patch, blue_patch])

    plt.plot(epsilon, fp_ratio, 'r', epsilon, fn_ratio, 'b', epsilon, cost, 'y')

    plt.xlabel('epsilon')
    plt.ylabel('False and True Negative Rate')

    x_linear = np.arange(0., 1.1, 0.1)

    plt.figure()
    plt.plot(fp_rate, tp_rate, 'b', x_linear, x_linear, 'r--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def draw_histogram_feature(x_data, x_header=None, y_value=None):
    m = len(x_data[0, :])
    for i in range(m):
        feature_value = x_data[:, i]

        if y_value is None:
            # the histogram of the data
            plt.hist(feature_value, 100, normed=1, facecolor='g', alpha=0.75)

            mean = np.mean(feature_value, axis=0)
            sigma = np.cov(feature_value, rowvar=False)

            num_bins = 50
            # the histogram of the data
            plt.hist(feature_value, num_bins,  facecolor='green', alpha=0.5)
            plt.xlabel('feature_value ' + str(i))
            plt.show()

            n, bins, patches = plt.hist(feature_value, num_bins, normed=1, facecolor='green',
                                        alpha=0.5)
            # add a 'best fit' line
            y = sts.norm.pdf(bins, mean, sigma)
            plt.plot(bins, y, 'r--')
            plt.show()

        else:

            plt.hist(feature_value, 50, normed=1, facecolor='g', alpha=0.75)
            n, bins, patches = plt.hist((y_value[:, i]), 50, normed=1, color='r', alpha=0.75)

            mean = np.mean(feature_value, axis=0)
            sigma = np.cov(feature_value, rowvar=False)

            y = sts.norm.pdf(bins, mean, sigma)
            plt.plot(bins, y, 'r--')
            plt.show()
