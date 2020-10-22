from semseg import constants
import json

OUTFILE_TEMPLATE = "{outdir}/{image_id}_{type}.json"


def format_report(metrics, invalid):
    """Given metrics as calculated by calc_report, formats into a dict.

    Parameters:
    metrics: dict with fields:
        precision: np array of length num_classes where precision[i] is the precision of class i.
        invalid_precision: np array of booleans, true where precision is invalid
        recall: same structure as precision but with recall values
        invalid_recall: same as invalid_precision for recall
        f1_score: same structure as precision but with f1 score values.
        invalid_f1_score: same as invalid_precision for f1_score
    invalid: value to populate invalid metrics with.

    Returns:
    dict where dict[class_name] = {"f1_score": f1_score, "precision": precision, "recall": recall}
    """

    result = {}

    for i in range(0, len(metrics['precision'])):
        result[constants.LABEL_CLASSES[i]] = {
            "f1_score": metrics['f1_score'][i] if ~metrics['invalid_f1_score'][i] else invalid,
            "precision": metrics['precision'][i] if ~metrics['invalid_precision'][i] else invalid,
            "recall": metrics['recall'][i] if ~metrics['invalid_recall'][i] else invalid,
        }

    return result


def write_report(outdir, image_id, type, contents):
    """Writes contents to file in  specified directory.
    """
    matrix_file = OUTFILE_TEMPLATE.format(outdir=outdir, image_id=image_id, type=type)
    with open(matrix_file, 'w') as f:
        f.write(contents)
        f.close()
