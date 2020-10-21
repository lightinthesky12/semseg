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

    # prepopulate matrix with zeros
    result = np.zeros((num_classes, num_classes))

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

    Parameters:
    matrix: confusion matrix, like that calculated by calc_confusion.

    Returns:
    precision: np array of length num_classes where precision[i] is the precision of class i.
    recall: same structure as precision but with recall values
    f1_score: same structure as precision but with f1 score values.

    """
    num_classes = len(matrix)
    assert matrix.shape == (num_classes, num_classes)

    total_positives = np.sum(matrix, 0)
    true_positives = [matrix[i][i] for i in range(0, num_classes)]
    total_labels = np.sum(matrix, 1)

    precision = np.zeros(num_classes)
    np.divide(true_positives, total_positives, out=precision, where=total_positives != 0)

    recall = np.zeros(num_classes)
    np.divide(true_positives, total_labels, out=recall, where=total_labels != 0)

    prec_plus_recall = precision + recall
    f1_score = np.zeros(num_classes)
    np.divide(precision * recall, prec_plus_recall, out=f1_score, where=prec_plus_recall != 0)
    f1_score = 2.0 * f1_score

    return precision, recall, f1_score
