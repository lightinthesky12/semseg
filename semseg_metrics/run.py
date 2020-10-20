from semseg_metrics import fetch_data, calc_metrics
import numpy as np

BASE_LABEL_URL = "https://storage.googleapis.com/aquarium-public/interview/semseg_metrics_coding_challenge/data/labels/{image_id}.png"
BASE_INFERENCE_URL = "https://storage.googleapis.com/aquarium-public/interview/semseg_metrics_coding_challenge/data/inferences/{image_id}.png"

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
    '000009_10'
]

def main():
    label = fetch_data.load_image_from_url(BASE_LABEL_URL.format(image_id=IMAGE_IDS[0]))
    inf = fetch_data.load_image_from_url(BASE_INFERENCE_URL.format(image_id=IMAGE_IDS[0]))

    matrix = calc_metrics.calc_confusion(label, inf)
    print(matrix)
    temp = calc_metrics.calc_report(matrix)
    print(temp)

if __name__ == "__main__":
    main()
