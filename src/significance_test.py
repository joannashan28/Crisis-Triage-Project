"""
Paired bootstrap test: is system A really better than system B on the same
test tweets, or could the gap be luck in which tweets ended up in the test set?

How it works:
  1. Line up the two systems' predictions on the same tweets.
  2. Many times (default 1,000): draw a random sample of tweets, with
     replacement, the same size as the test set. Score both systems on
     that sample and record the difference (A minus B).
  3. The 95% confidence interval is the middle 95% of those differences.
     The p-value is the share of samples where A did NOT beat B.

Works on any two predictions.tsv files: a stage run (columns gold / pred) or
an end-to-end run (columns gold_end_to_end / pred_end_to_end).

Usage:
    python src/significance_test.py \
        --a runs/bertweet-base/stage2/seed1/predictions.tsv \
        --b runs/tfidf_lr/stage2/seed0/predictions.tsv \
        --metric macro_f1
"""

import argparse
import random

from sklearn.metrics import precision_recall_fscore_support

from crisisbench_io import read_tsv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="predictions.tsv of system A")
    parser.add_argument("--b", required=True, help="predictions.tsv of system B")
    parser.add_argument("--metric", default="weighted_f1",
                        choices=["weighted_f1", "macro_f1"])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def gold_and_pred_columns(row):
    if "gold_end_to_end" in row:
        return "gold_end_to_end", "pred_end_to_end"
    return "gold", "pred"


def load_by_id(path):
    rows = read_tsv(path)
    by_id = {}
    for row in rows:
        by_id[row["id"]] = row
    return by_id


def score(gold, predicted, labels, metric):
    if metric == "weighted_f1":
        average = "weighted"
    else:
        average = "macro"
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=labels, average=average, zero_division=0)
    return float(f1)


def main():
    args = parse_args()
    a_by_id = load_by_id(args.a)
    b_by_id = load_by_id(args.b)

    shared_ids = []
    for tweet_id in a_by_id:
        if tweet_id in b_by_id:
            shared_ids.append(tweet_id)
    shared_ids.sort()

    if len(shared_ids) == 0:
        raise RuntimeError("The two files have no tweets in common.")

    first_a = a_by_id[shared_ids[0]]
    gold_column, pred_column = gold_and_pred_columns(first_a)

    gold = []
    a_predicted = []
    b_predicted = []
    for tweet_id in shared_ids:
        gold.append(a_by_id[tweet_id][gold_column])
        a_predicted.append(a_by_id[tweet_id][pred_column])
        b_predicted.append(b_by_id[tweet_id][pred_column])

    label_set = set()
    label_set.update(gold)
    label_set.update(a_predicted)
    label_set.update(b_predicted)
    labels = sorted(label_set)

    observed_a = score(gold, a_predicted, labels, args.metric)
    observed_b = score(gold, b_predicted, labels, args.metric)
    observed_difference = observed_a - observed_b

    sampler = random.Random(args.seed)
    count = len(shared_ids)
    differences = []
    a_not_better = 0

    for sample_number in range(args.samples):
        sample_gold = []
        sample_a = []
        sample_b = []
        for draw in range(count):
            index = sampler.randrange(count)
            sample_gold.append(gold[index])
            sample_a.append(a_predicted[index])
            sample_b.append(b_predicted[index])

        difference = score(sample_gold, sample_a, labels, args.metric) - \
            score(sample_gold, sample_b, labels, args.metric)
        differences.append(difference)
        if difference <= 0:
            a_not_better += 1

    differences.sort()
    lower = differences[int(0.025 * args.samples)]
    upper = differences[int(0.975 * args.samples) - 1]
    p_value = a_not_better / args.samples

    print("Tweets compared: " + str(count))
    print("%s  A = %.4f   B = %.4f   A - B = %.4f" % (
        args.metric, observed_a, observed_b, observed_difference))
    print("95%% bootstrap interval for A - B: [%.4f, %.4f]" % (lower, upper))
    print("p-value (share of samples where A did not beat B): %.4f" % p_value)
    if p_value < 0.05:
        print("A is better than B at the 0.05 level.")
    else:
        print("Not significant at the 0.05 level.")


if __name__ == "__main__":
    main()
