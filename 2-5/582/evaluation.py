import argparse


def read_prediction(prediction_file):
    return read_strings(prediction_file)


def read_ground_truth(ground_truth_file):
    return read_strings(ground_truth_file)


def f1_score(gt, pred):
    if len(pred) == 0:
        return 0.0
    intsct_len = len(set(gt).intersection(set(pred)))
    if intsct_len == 0:
        return 0.0
    precision = intsct_len / len(pred)
    recall = intsct_len / len(gt)
    return 2. / (1. / precision + 1. / recall)


def evaluation_metrics(prediction_file: str, ground_truth_file: str):
    while True:
        # prediction, ground_truth
        # example : [('A', ['a', 'b', 'c']), ('B', ['d', 'e']), ('C', ['f', 'g', 'h'])]
        prediction = read_prediction(prediction_file)
        ground_truth = read_ground_truth(ground_truth_file)

        pred_dict = dict(prediction)
        f1_sum = 0.0

        for query, match in ground_truth:
            if query in pred_dict:
                pred_match = pred_dict[query]
                f1_sum += f1_score(match, pred_match)

        mean_f1 = f1_sum / len(ground_truth)
        break


    return mean_f1


def read_strings(input_file):
    lines = open(input_file, "r").read().splitlines()
    query_matches = [line.split(' ') for line in lines]
    return [(query, matches.split(',')) for query, matches in query_matches]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=str, default='pred.txt')
    parser.add_argument('--test_label_path', type=str)
    args = parser.parse_args()

    print(evaluation_metrics(args.prediction, args.test_label_path))