import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mptches
import seaborn as sns


def draw_correlation_matrix(sigma, data):

    # Get correlation matrix and draw it
    corr_matrix = np.corrcoef(sigma)
    features = data.columns.values.tolist()
    # print features
    sns.set(style="white")
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=features, yticklabels=features)

    plt.xlabel('Correlation Matrix')
    plt.show()

    # sns.pairplot(data[['area', 'perimeterPercent', 'ridgePercent']])
    # sns.pairplot(data[['area', 'perimeterPercent', 'ridgePercent', 'hipPercent',
    # 'valleyPercent']])
    # plt.show()


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
    # plt.plot(epsilon, fp_ratio, 'r', epsilon, fn_ratio, 'b')

    # plt.plot(epsilon, fp_ratio, 'r')
    # plt.plot(epsilon, fn_ratio, 'b')

    plt.xlabel('epsilon')
    plt.ylabel('False and True Negative Rate')

    # plt.show()
    x_linear = np.arange(0., 1.1, 0.1)

    plt.figure()
    plt.plot(fp_rate, tp_rate, 'b', x_linear, x_linear, 'r--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return


def draw_histogram_feature(x_data, x_header=None, y_value=None):
    m = len(x_data[0, :])
    for i in range(m):
        feature_value = x_data[:, i]
        if x_header is not None:
            print(x_header[i])
        else:
            print(i)
        if y_value is None:
            # the histogram of the data
            n, bins, patches = plt.hist(feature_value, 100, normed=1, facecolor='g', alpha=0.75)
            feature_value1 = np.log(feature_value + 1)

            # feature_value2 = 2*np.power(feature_value, -2)
            # feature_value2 = np.power(feature_value, 0.5)
            # feature_value2 = np.power(feature_value, -0.5)
            feature_value2 = np.power(feature_value, 0.2)
            # feature_value = np.power(feature_value, 2) + 10
            # feature_value = np.log(feature_value + 5)
            # feature_value = np.power(feature_value, 0.1)
            # example data2
            mean = np.mean(feature_value, axis=0)
            sigma = np.cov(feature_value, rowvar=0)
            mean1 = np.mean(feature_value1, axis=0)
            sigma1 = np.cov(feature_value1, rowvar=0)
            mean2 = np.mean(feature_value2, axis=0)
            sigma2 = np.cov(feature_value2, rowvar=0)
            num_bins = 50
            # the histogram of the data
            # n, bins, patches = plt.hist(feature_value, num_bins,  facecolor='green', alpha=0.5)
            # plt.xlabel('feature_value ' + str(i))
            # plt.show()

            n, bins, patches = plt.hist(feature_value, num_bins, normed=1, facecolor='green', alpha=0.5)
            # add a 'best fit' line
            y = plt.mlab.normpdf(bins, mean, sigma)
            plt.plot(bins, y, 'r--')
            plt.show()

            n, bins, patches = plt.hist(feature_value1, num_bins, normed=1, facecolor='green', alpha=0.5)
            # add a 'best fit' line
            y = plt.mlab.normpdf(bins, mean1, sigma1)
            plt.plot(bins, y, 'r--')
            plt.show()

            n, bins, patches = plt.hist(feature_value2, num_bins, normed=1, facecolor='green', alpha=0.5)
            # add a 'best fit' line
            y = plt.mlab.normpdf(bins, mean2, sigma2)
            plt.plot(bins, y, 'r--')
            plt.show()

        else:

            n, bins, patches = plt.hist(feature_value, 50, normed=1, facecolor='g', alpha=0.75)
            n, bins, patches = plt.hist((y_value[:, i]), 50, normed=1, color='r', alpha=0.75)

            y = plt.mlab.normpdf(bins, mean2, sigma2)
            plt.plot(bins, y, 'r--')
            plt.show()
        # plt.xlabel('Feature')
        # plt.ylabel('Probability')
        # plt.title('Histogram')
        # plt.grid(True)
        # plt.show()

    return