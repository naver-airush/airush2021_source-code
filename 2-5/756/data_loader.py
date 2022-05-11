import os

from nsml.constants import DATASET_PATH


def test_data_loader(root_path):
    return root_path


def write_output(output_file, data):
    with open(output_file, 'w') as f:
        for x in data:
            f.write(x[0] + ' ' + ','.join(x[1]) + '\n')


def feed_infer(output_file, infer_func):
    print('DATASET_PATH=', DATASET_PATH)
    #os.system('/bin/ls -lR ' + DATASET_PATH)
    prediction = infer_func(test_data_loader(DATASET_PATH))
    write_output(output_file, prediction)
    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')

