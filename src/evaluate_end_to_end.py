"""
Score the full two-stage pipeline on the end-to-end test set (tweets that
appear in both CrisisBench test sets, so each has a gold label for both
stages).

How a tweet's pipeline output is decided:
  - If Stage 1 says P(informative) < threshold, the tweet is filtered out
    and the pipeline output is "not actionable".
  - Otherwise, Stage 2's category is the output. If Stage 2 says
    "not humanitarian", that also counts as "not actionable".

This is scored against gold_end_to_end from prepare_data.py.

Besides the usual metrics, this reports:
  - For each humanitarian category: how many gold tweets were caught,
    and how many were lost because Stage 1 filtered them out. This is the
    "urgent cries for help filtered as noise" risk from our proposal.
  - A threshold sweep: how the results change as the Stage 1 cutoff moves,
    showing the trade-off between missing real reports and letting noise in.
  - Breakdowns by disaster type, source dataset, and tweet length.

Stage 1 and Stage 2 predictions come from each run's predictions.tsv, so you
can mix models (e.g. BERTweet Stage 1 with a TF-IDF Stage 2).

Usage:
    python src/evaluate_end_to_end.py --stage1-model bertweet-base --stage1-seed 1 \
                                      --stage2-model bertweet-base --stage2-seed 1
    python src/evaluate_end_to_end.py --stage1-model tfidf_lr --stage1-seed 0 \
                                      --stage2-model tfidf_lr --stage2-seed 0
"""

import argparse
import os

import config
from crisisbench_io import read_tsv, save_json, write_tsv
from metrics import compute_metrics, metrics_by_group, save_confusion_matrix_plot
from run_outputs import run_directory


THRESHOLDS_TO_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

OUTPUT_COLUMNS = [
    "id",
    "gold_end_to_end",
    "pred_end_to_end",
    "error_type",
    "gold_informativeness",
    "gold_humanitarian",
    "stage1_prob_informative",
    "stage2_pred",
    "stage2_confidence",
    "disaster_type",
    "source",
    "event",
    "length_bucket",
    "text",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-model", required=True)
    parser.add_argument("--stage1-seed", type=int, required=True)
    parser.add_argument("--stage2-model", required=True)
    parser.add_argument("--stage2-seed", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_STAGE1_THRESHOLD)
    parser.add_argument("--data-dir", default=config.PROCESSED_DATA_DIR)
    return parser.parse_args()


def load_predictions_by_id(model, stage, seed):
    run_dir = run_directory(model, stage, seed)
    path = os.path.join(run_dir, "predictions.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(path + " does not exist. Train that model first.")

    rows = read_tsv(path)
    by_id = {}
    for row in rows:
        by_id[row["id"]] = row
    return by_id


def pipeline_output(stage1_probability, stage2_label, threshold):
    """The pipeline's final label for one tweet, and whether Stage 1
    filtered it out."""
    if stage1_probability < threshold:
        return config.NOT_ACTIONABLE_LABEL, True
    if stage2_label == config.NOT_HUMANITARIAN_LABEL:
        return config.NOT_ACTIONABLE_LABEL, False
    return stage2_label, False


def describe_error(gold, predicted, filtered_by_stage1, stage2_label):
    """Name the kind of mistake, so errors can be counted and reviewed by type."""
    if gold == predicted:
        return "correct"

    if gold == config.NOT_ACTIONABLE_LABEL:
        return "noise_passed_through"

    # From here on, the gold label is a real humanitarian category.
    if filtered_by_stage1:
        return "stage1_filtered_actionable"
    if stage2_label == config.NOT_HUMANITARIAN_LABEL:
        return "stage2_marked_other"
    return "stage2_wrong_category"


def run_pipeline(end_to_end_rows, stage1_by_id, stage2_by_id, threshold):
    """Apply the pipeline at a given threshold. Returns predicted labels and
    a matching list of 'filtered by Stage 1' flags."""
    predicted = []
    filtered_flags = []
    for row in end_to_end_rows:
        tweet_id = row["id"]
        stage1_probability = float(stage1_by_id[tweet_id]["prob_informative"])
        stage2_label = stage2_by_id[tweet_id]["pred"]
        label, filtered = pipeline_output(stage1_probability, stage2_label, threshold)
        predicted.append(label)
        filtered_flags.append(filtered)
    return predicted, filtered_flags


def category_recall_report(gold_labels, predicted_labels, filtered_flags):
    """For each real humanitarian category: how many gold tweets there were,
    how many the pipeline labeled correctly, and how many Stage 1 threw away."""
    report = {}
    for index in range(len(gold_labels)):
        gold = gold_labels[index]
        if gold == config.NOT_ACTIONABLE_LABEL:
            continue

        if gold not in report:
            report[gold] = {"n": 0, "correct": 0, "filtered_by_stage1": 0}

        report[gold]["n"] += 1
        if predicted_labels[index] == gold:
            report[gold]["correct"] += 1
        if filtered_flags[index]:
            report[gold]["filtered_by_stage1"] += 1

    for label in report:
        entry = report[label]
        entry["recall"] = entry["correct"] / entry["n"]
        entry["share_filtered_by_stage1"] = entry["filtered_by_stage1"] / entry["n"]

    return report


def triage_rates(gold_labels, predicted_labels, filtered_flags):
    """Two numbers that matter for responders:
      - share of actionable tweets that Stage 1 filtered out (missed reports)
      - share of not-actionable tweets that ended up in a category (noise)"""
    actionable_total = 0
    actionable_filtered = 0
    noise_total = 0
    noise_passed = 0

    for index in range(len(gold_labels)):
        gold = gold_labels[index]
        if gold == config.NOT_ACTIONABLE_LABEL:
            noise_total += 1
            if predicted_labels[index] != config.NOT_ACTIONABLE_LABEL:
                noise_passed += 1
        else:
            actionable_total += 1
            if filtered_flags[index]:
                actionable_filtered += 1

    rates = {}
    if actionable_total > 0:
        rates["actionable_filtered_by_stage1"] = actionable_filtered / actionable_total
    if noise_total > 0:
        rates["noise_placed_in_a_category"] = noise_passed / noise_total
    return rates


def output_folder_name(args):
    name = ("stage1-" + args.stage1_model + "-seed" + str(args.stage1_seed)
            + "__stage2-" + args.stage2_model + "-seed" + str(args.stage2_seed))
    if args.threshold != config.DEFAULT_STAGE1_THRESHOLD:
        name = name + "__threshold" + str(args.threshold)
    return name


def main():
    args = parse_args()

    end_to_end_path = os.path.join(args.data_dir, "end_to_end_test.tsv")
    end_to_end_rows = read_tsv(end_to_end_path)

    stage1_by_id = load_predictions_by_id(args.stage1_model, 1, args.stage1_seed)
    stage2_by_id = load_predictions_by_id(args.stage2_model, 2, args.stage2_seed)

    # Every end-to-end tweet is in both test sets, so it must have both predictions.
    for row in end_to_end_rows:
        if row["id"] not in stage1_by_id or row["id"] not in stage2_by_id:
            raise RuntimeError("Tweet " + row["id"] + " is missing a prediction. "
                               "Were the models trained on the current data/processed files?")

    gold_labels = []
    for row in end_to_end_rows:
        gold_labels.append(row["gold_end_to_end"])

    predicted, filtered_flags = run_pipeline(end_to_end_rows, stage1_by_id,
                                             stage2_by_id, args.threshold)

    label_set = set()
    label_set.update(gold_labels)
    label_set.update(predicted)
    labels_in_order = sorted(label_set)

    # Main metrics
    metrics = compute_metrics(gold_labels, predicted, labels_in_order)
    metrics["system"] = args.stage1_model + " -> " + args.stage2_model
    metrics["stage1_model"] = args.stage1_model
    metrics["stage1_seed"] = args.stage1_seed
    metrics["stage2_model"] = args.stage2_model
    metrics["stage2_seed"] = args.stage2_seed
    metrics["threshold"] = args.threshold
    metrics["is_end_to_end"] = True
    metrics["category_recall"] = category_recall_report(gold_labels, predicted, filtered_flags)
    metrics["triage_rates"] = triage_rates(gold_labels, predicted, filtered_flags)

    metrics["by_disaster_type"] = metrics_by_group(end_to_end_rows, gold_labels, predicted,
                                                   "disaster_type")
    metrics["by_source"] = metrics_by_group(end_to_end_rows, gold_labels, predicted, "source")
    metrics["by_length"] = metrics_by_group(end_to_end_rows, gold_labels, predicted,
                                            "length_bucket")

    # Threshold sweep
    sweep = []
    for threshold in THRESHOLDS_TO_SWEEP:
        sweep_predicted, sweep_filtered = run_pipeline(end_to_end_rows, stage1_by_id,
                                                       stage2_by_id, threshold)
        sweep_metrics = compute_metrics(gold_labels, sweep_predicted, labels_in_order)
        rates = triage_rates(gold_labels, sweep_predicted, sweep_filtered)
        sweep_entry = {
            "threshold": threshold,
            "weighted_f1": sweep_metrics["weighted_f1"],
            "macro_f1": sweep_metrics["macro_f1"],
        }
        sweep_entry.update(rates)
        sweep.append(sweep_entry)
    metrics["threshold_sweep"] = sweep

    # Save everything
    output_dir = os.path.join(config.RUNS_DIR, "end_to_end", output_folder_name(args))
    os.makedirs(output_dir, exist_ok=True)

    output_rows = []
    for index in range(len(end_to_end_rows)):
        row = end_to_end_rows[index]
        stage1_row = stage1_by_id[row["id"]]
        stage2_row = stage2_by_id[row["id"]]
        error_type = describe_error(gold_labels[index], predicted[index],
                                    filtered_flags[index], stage2_row["pred"])
        output_row = dict(row)
        output_row["pred_end_to_end"] = predicted[index]
        output_row["error_type"] = error_type
        output_row["stage1_prob_informative"] = stage1_row["prob_informative"]
        output_row["stage2_pred"] = stage2_row["pred"]
        output_row["stage2_confidence"] = stage2_row["confidence"]
        output_rows.append(output_row)

    write_tsv(os.path.join(output_dir, "predictions.tsv"), output_rows, OUTPUT_COLUMNS)
    save_json(metrics, os.path.join(output_dir, "metrics.json"))
    save_confusion_matrix_plot(metrics, os.path.join(output_dir, "confusion_matrix.png"),
                               "End to end: " + metrics["system"])

    print("End-to-end results for " + metrics["system"] + " (threshold "
          + str(args.threshold) + ", " + str(len(end_to_end_rows)) + " tweets):")
    print("  weighted F1 = %.4f   macro F1 = %.4f" % (metrics["weighted_f1"],
                                                     metrics["macro_f1"]))
    rates = metrics["triage_rates"]
    if "actionable_filtered_by_stage1" in rates:
        print("  actionable tweets filtered out by Stage 1: %.1f%%" % (
            100 * rates["actionable_filtered_by_stage1"]))
    if "noise_placed_in_a_category" in rates:
        print("  not-actionable tweets placed in a category: %.1f%%" % (
            100 * rates["noise_placed_in_a_category"]))
    print("  saved to " + output_dir)


if __name__ == "__main__":
    main()
