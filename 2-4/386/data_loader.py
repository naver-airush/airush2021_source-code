import os

from nsml import DATASET_PATH


def read_strings(input_file):
    return open(input_file, "r").read().splitlines()


def write_strings(output_file, data):
    with open(output_file, "w") as f:
        for x in data:
            f.write(str(x) + "\n")


def test_data_loader(root_path):
    return read_strings(os.path.join(root_path, 'test', 'test_data'))


def feed_infer(output_file, infer_func):
    prediciton = infer_func(test_data_loader(DATASET_PATH))
    print('write output')
    write_strings(output_file, prediciton)
    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')
