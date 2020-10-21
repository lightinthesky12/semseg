from semseg import fetch_data, calc_metrics, write_report, constants
import numpy as np


def main():
    label = fetch_data.load_image_from_url(constants.BASE_LABEL_URL.format(image_id=constants.IMAGE_IDS[0]))
    np.savetxt('label.txt', label, fmt='%i')
    inf = fetch_data.load_image_from_url(constants.BASE_INFERENCE_URL.format(image_id=constants.IMAGE_IDS[0]))
    print(label)

    matrix = calc_metrics.calc_confusion(label, inf, len(constants.LABEL_CLASSES))
    precision, recall, f1_score = calc_metrics.calc_report(matrix)
    report = write_report.format_report(precision, recall, f1_score)
    print(report)


if __name__ == "__main__":
    main()
