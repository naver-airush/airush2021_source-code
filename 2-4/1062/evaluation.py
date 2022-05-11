import argparse

from nltk.translate.gleu_score import corpus_gleu


def read_strings(input_file):
    return open(input_file, "r").read().splitlines()


def read_prediction(prediction_file):
    return read_strings(prediction_file)


def read_ground_truth(ground_truth_file):
    return read_strings(ground_truth_file)


def em(prediction, ground_truth):
    return sum([x == y for x, y in zip(prediction, ground_truth)]) / len(ground_truth) * 100.


def gleu(prediction, ground_truth):
    return corpus_gleu([[x] for x in ground_truth], prediction) * 100.


def evaluation_metrics(prediction_file: str, ground_truth_file: str):
    try:
        prediction = read_prediction(prediction_file)
        ground_truth = read_ground_truth(ground_truth_file)
        score = gleu(prediction, ground_truth)
    except:
        score = 0.0
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=str, default='pred.txt')
    parser.add_argument('--test_label_path', type=str)
    args = parser.parse_args()

    print(evaluation_metrics(args.prediction, args.test_label_path))
