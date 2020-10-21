LABEL_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle"
]

IGNORED_PIXEL = 255

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
    '000009_10',
]