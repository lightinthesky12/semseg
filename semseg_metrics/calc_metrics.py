import numpy as np
from semseg_metrics import constants


def calc_confusion(labels, inference):
    num_classes = len(constants.LABEL_CLASSES)
    result = [[0 for i in range(0, num_classes)]  for j in range(0, num_classes)]

    for x, row in np.ndenumerate(labels):
        for y, label in np.ndenumerate(row):
            inf = inference[x][y]

            if (label != constants.IGNORED_PIXEL) and (inf != constants.IGNORED_PIXEL):
                result[label][inf] += 1

    return result

def calc_report(matrix):
    num_classes = len(constants.LABEL_CLASSES)
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

    result = {}

    for i in range(0, num_classes):
        result[constants.LABEL_CLASSES[i]] = {"f1_score": f1_score[i], "precision": precision[i], "recall": recall[i]}

    return result