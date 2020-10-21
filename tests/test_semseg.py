from semseg import fetch_data, calc_metrics, write_report
from semseg import constants
import numpy as np

CONFUSION_MATRIX = [[114935, 284, 0, 0, 0, 0, 0, 0, 484, 16, 0, 0, 0, 342, 0, 0, 0, 0, 0], [1120, 4968, 0, 189, 115, 38, 0, 12, 469, 72, 0, 0, 0, 806, 0, 0, 0, 0, 0], [7, 0, 93752, 0, 1022, 72, 0, 0, 3040, 0, 80, 0, 0, 29, 0, 0, 0, 0, 0], [0, 102, 0, 3350, 509, 0, 0, 55, 76, 0, 0, 0, 0, 518, 0, 0, 0, 0, 0], [0, 0, 242, 185, 16922, 6, 0, 0, 361, 0, 0, 0, 0, 37, 0, 0, 0, 0, 0], [0, 11, 295, 0, 325, 802, 0, 0, 203, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 56, 194, 20, 0, 0, 1273, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [50, 542, 3012, 34, 238, 49, 0, 2, 111610, 195, 226, 0, 0, 804, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 176, 0, 7, 0, 0, 0, 122, 0, 14542, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 209, 0, 1, 0, 0, 0, 137, 0, 0, 0, 0, 166, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [613, 48, 56, 9, 107, 0, 0, 0, 489, 36, 0, 0, 0, 84032, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 51, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
REPORT = {"road": {"f1_score": 0.9874734734906738, "precision": 0.9846648104519169, "recall": 0.9902982052541336}, "sidewalk": {"f1_score": 0.7229336437718277, "precision": 0.834256926952141, "recall": 0.6378225702914366}, "building": {"f1_score": 0.9576155625807572, "precision": 0.958599605321009, "recall": 0.9566335380910593}, "wall": {"f1_score": 0.7817057519542644, "precision": 0.845746023731381, "recall": 0.7266811279826464}, "fence": {"f1_score": 0.9142332315837813, "precision": 0.8783348904806395, "recall": 0.9531910099701459}, "pole": {"f1_score": 0.6162120630042259, "precision": 0.8293691830403309, "recall": 0.4902200488997555}, "traffic_light": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}, "traffic_sign": {"f1_score": 0.8767217630853994, "precision": 0.9485842026825634, "recall": 0.8149807938540333}, "vegetation": {"f1_score": 0.954474124181693, "precision": 0.9530762990478631, "recall": 0.9558760555660232}, "terrain": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}, "sky": {"f1_score": 0.9794241454790369, "precision": 0.9793911637931034, "recall": 0.979457129386408}, "person": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}, "rider": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}, "car": {"f1_score": 0.9763896635062279, "precision": 0.9688026009361526, "recall": 0.9840964984190186}, "truck": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}, "bus": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}, "train": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}, "motorcycle": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}, "bicycle": {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}}


def test_load_label_from_url():
    label = fetch_data.load_image_from_url(constants.BASE_LABEL_URL.format(image_id=constants.IMAGE_IDS[0]))
    true_label = np.loadtxt('tests/data/label.txt', dtype=int)
    assert np.array_equal(label, true_label)


def test_load_inference_from_url():
    inf = fetch_data.load_image_from_url(constants.BASE_INFERENCE_URL.format(image_id=constants.IMAGE_IDS[0]))
    true_inf = np.loadtxt('tests/data/inferences.txt', dtype=int)
    assert np.array_equal(inf, true_inf)


def test_confusion():
    label = np.loadtxt('tests/data/label.txt', dtype=int)
    inf = np.loadtxt('tests/data/inferences.txt', dtype=int)
    confusion = calc_metrics.calc_confusion(label, inf, len(constants.LABEL_CLASSES))
    assert np.array_equal(confusion, np.asarray(CONFUSION_MATRIX))


def test_report():
    precision, recall, f1_score = calc_metrics.calc_report(CONFUSION_MATRIX)
    report = write_report.format_report(precision, recall, f1_score)
    assert report == REPORT