from semseg import constants


def format_report(precision, recall, f1_score):
    """Given metrics as calculated by calc_report, formats into a dict.

    Parameters:
    precision: np array of length num_classes where precision[i] is the precision of class i.
    recall: same structure as precision but with recall values
    f1_score: same structure as precision but with f1 score values.

    Returns:
    dict where dict[class_name] = {"f1_score": f1_score, "precision": precision, "recall": recall}
    """

    result = {}

    for i in range(0, len(precision)):
        result[constants.LABEL_CLASSES[i]] = {"f1_score": f1_score[i], "precision": precision[i], "recall": recall[i]}

    return result
