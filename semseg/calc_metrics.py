import numpy as np
from semseg import constants


def calc_confusion(labels, inference, num_classes):
    """Given labels and inferences for an image, calculates the confusion matrix.
    Assumes labels and inferences are the same shape.

    Parameters:
    labels: numpy array where the value of each element represents the label class of the pixel.
    inference: numpy array where the value of each element represents the inference class of the pixel.
    num_classes: number of classes considered

    Returns:
    numpy array of dimension num_classes x num_classes where element at matrix[i][j] represents the number
    of labels of class i were predicted as class j.

    """
    assert (labels.shape == inference.shape)
    assert np.all((labels < num_classes) | (labels == 255))
    assert np.all((inference < num_classes) | (inference == 255))

    # prepopulate matrix with zeros
    result = np.zeros((num_classes, num_classes), dtype=int)

    for x, row in np.ndenumerate(labels):
        for y, label in np.ndenumerate(row):
            inf = inference[x][y]

            # if pixel should be ignored, skip. otherwise, increment value in the matrix
            if (label != constants.IGNORED_PIXEL) and (inf != constants.IGNORED_PIXEL):
                result[label][inf] += 1

    return result


def calc_report(matrix):
    """Given a confusion matrix, calculate precision, recall, and f1 score for each class.
    Assumes confusion matrix is square of dimension num_classes x num_classes.
    precision: TruePositives / (TruePositives + FalsePositives)
    recall: TruePositives / (TruePositives + FalseNegatives)
    f1_score: 2.0 * ((precision * recall) / (precision + recall))

    Parameters:
    matrix: confusion matrix, like that calculated by calc_confusion. returns -1 for invalid values.

    Returns:
    metrics: dict with fields:
        precision: np array of length num_classes where precision[i] is the precision of class i.
        invalid_precision: np array of booleans, true where precision is invalid
        recall: same structure as precision but with recall values
        invalid_recall: same as invalid_precision for recall
        f1_score: same structure as precision but with f1 score values.
        invalid_f1_score: same as invalid_precision for f1_score

    """
    num_classes = len(matrix)
    assert matrix.shape == (num_classes, num_classes)

    # calculates total positives for each class by summing over 0 axis (result[j] = sum[i][j] over all i)
    total_positives = np.sum(matrix, 0)
    # true positives for each class
    true_positives = [matrix[i][i] for i in range(0, num_classes)]
    # calculates total labels for each class by summing over 1 axis (result[i] = sum[i][j] over all j)
    total_labels = np.sum(matrix, 1)

    # flag where precision is invalid due to lack of data
    invalid_precision = total_positives == 0
    precision = np.divide(true_positives, total_positives, where=~invalid_precision)

    # flag where recall is invalid due to lack of data
    invalid_recall = total_labels == 0
    recall = np.divide(true_positives, total_labels, where=~invalid_recall)

    prec_plus_recall = precision + recall
    # flag where f1 is invalid (where precision or recall is invalid, or precision + recall is 0
    invalid_f1_score = invalid_precision | invalid_recall | (prec_plus_recall == 0)
    f1_score = np.divide(2.0 * precision * recall, prec_plus_recall, where=~invalid_f1_score)

    return {
        "precision": precision,
        "invalid_precision": invalid_precision,
        "recall": recall,
        "invalid_recall": invalid_recall,
        "f1_score": f1_score,
        "invalid_f1_score": invalid_f1_score
    }
