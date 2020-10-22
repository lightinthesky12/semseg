# Semantic Segmentation Script
## Running
This script is designed to write a confusion matrix file and metrics report file for each of the ten images, as well as a combined report, to an output directory.


To run the script, from the root directory, run:

1. `source venv/bin/activate`
2. `python main.py --outdir <OUTPUT_DIRECTORY>`

The script also takes two additional optional arguments:

`--image <IMAGE>` to run script on a single specific image

`--invalid <INVALID>` to specify how invalid metrics should be represented. Default is None.

Labels and inferences are assumed to be the dimensions of the image, where the value of each pixel represents the class assigned to that pixel. Pixels of value 255 are skipped.


## Development

Additional documentation can be viewed by running `pydoc -b` in the root directory. All logic is located under `semseg`.
Tests are located in `tests/` can be run with `pytest` in the root directory.