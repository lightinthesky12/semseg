from semseg import fetch_data, calc_metrics, write_report, constants
import numpy as np
import argparse
import json


IMAGE_IDS = [
    '000000_10',
    '000001_10',
    '000002_10',
    '000003_10',
    '000004_10',
    '000005_10',
    '000006_10',
    '000007_10',
    '000008_10',
    '000009_10',
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--outdir', help='Directory to write the output', required=True)
    parser.add_argument('--image', help='Image ID to process')
    parser.add_argument('--invalid', help="Value to set invalid metrics. Defaults to None.", default=None)

    args = parser.parse_args()

    cumulative_matrix = None

    images = IMAGE_IDS
    if args.image is not None:
        images = [args.image]

    for image_id in images:
        label = fetch_data.load_image_from_url(constants.BASE_LABEL_URL.format(image_id=image_id))
        inf = fetch_data.load_image_from_url(constants.BASE_INFERENCE_URL.format(image_id=image_id))

        matrix = calc_metrics.calc_confusion(label, inf, len(constants.LABEL_CLASSES))

        # update cumulateive matrix
        if cumulative_matrix is None:
            cumulative_matrix = matrix
        else:
            cumulative_matrix += matrix

        metrics = calc_metrics.calc_report(matrix)
        report = write_report.format_report(metrics, args.invalid)

        write_report.write_report(args.outdir, image_id, "matrix", json.dumps(matrix.tolist()))
        write_report.write_report(args.outdir, image_id, "report", json.dumps(report))

    metrics = calc_metrics.calc_report(cumulative_matrix)
    report = write_report.format_report(metrics, args.invalid)

    write_report.write_report(args.outdir, "cumulative", "matrix", json.dumps(cumulative_matrix.tolist()))
    write_report.write_report(args.outdir, "cumulative", "report", json.dumps(report))


if __name__ == "__main__":
    main()
