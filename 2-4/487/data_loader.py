"""This file is not really used"""

import os

from nsml import DATASET_PATH
from utils.utils import read_strings, write_strings


def test_data_loader(root_path):
    return read_strings(os.path.join(root_path, 'test', 'test_data'))


def feed_infer(output_file, infer_func):
    prediciton = infer_func(test_data_loader(DATASET_PATH))
    print('write output')
    write_strings(output_file, prediciton)
    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')
