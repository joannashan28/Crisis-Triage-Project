"""
Saving the results of one run (one model, one stage, one seed).

Every model (majority baseline, TF-IDF, BERTweet, RoBERTa) saves its output
through save_run, so all runs have identical files:

  runs/<model>/stage<1|2>/seed<N>/
      predictions.tsv       one row per test tweet: gold, predicted, confidence
      metrics.json          overall, per-class, and per-group results
      confusion_matrix.png
      model/                the saved model (used by the demo server)
"""

import os

import config
from crisisbench_io import length_bucket, save_json, write_tsv
from metrics import compute_metrics, metrics_by_group, save_confusion_matrix_plot


PREDICTION_COLUMNS = [
    "id",
    "gold",
    "pred",
    "confidence",
    "prob_informative",
    "disaster_type",
    "source",
    "event",
    "text",
]


def run_directory(model_short_name, stage, seed):
    folder_name = "stage" + str(stage)
    seed_name = "seed" + str(seed)
    return os.path.join(config.RUNS_DIR, model_short_name, folder_name, seed_name)


def save_run(run_dir, model_short_name, stage, seed, test_rows, predicted_labels,
             confidences, informative_probabilities, labels_in_order, run_settings):
    """Write predictions, metrics, and a confusion matrix for one run.

    informative_probabilities is P(informative) for each test tweet in
    Stage 1 runs. Stage 2 runs pass None. The end-to-end evaluation uses
    these probabilities to try different Stage 1 thresholds.
    """
    os.makedirs(run_dir, exist_ok=True)

    gold_labels = []
    for row in test_rows:
        gold_labels.append(row["label"])

    # predictions.tsv
    prediction_rows = []
    for index in range(len(test_rows)):
        row = test_rows[index]
        if informative_probabilities is None:
            informative_probability = ""
        else:
            informative_probability = "%.6f" % informative_probabilities[index]

        prediction_row = {
            "id": row["id"],
            "gold": row["label"],
            "pred": predicted_labels[index],
            "confidence": "%.6f" % confidences[index],
            "prob_informative": informative_probability,
            "disaster_type": row["disaster_type"],
            "source": row["source"],
            "event": row["event"],
            "text": row["text"],
        }
        prediction_rows.append(prediction_row)
    write_tsv(os.path.join(run_dir, "predictions.tsv"), prediction_rows, PREDICTION_COLUMNS)

    # metrics.json
    metrics = compute_metrics(gold_labels, predicted_labels, labels_in_order)
    metrics["model"] = model_short_name
    metrics["stage"] = stage
    metrics["seed"] = seed
    metrics["settings"] = run_settings

    rows_with_length = []
    for row in test_rows:
        row_copy = dict(row)
        row_copy["length_bucket"] = length_bucket(row["text"])
        rows_with_length.append(row_copy)

    metrics["by_disaster_type"] = metrics_by_group(
        rows_with_length, gold_labels, predicted_labels, "disaster_type")
    metrics["by_source"] = metrics_by_group(
        rows_with_length, gold_labels, predicted_labels, "source")
    metrics["by_length"] = metrics_by_group(
        rows_with_length, gold_labels, predicted_labels, "length_bucket")

    save_json(metrics, os.path.join(run_dir, "metrics.json"))

    # confusion_matrix.png
    title = model_short_name + ", Stage " + str(stage) + ", seed " + str(seed)
    save_confusion_matrix_plot(metrics, os.path.join(run_dir, "confusion_matrix.png"), title)

    print("Stage " + str(stage) + " test results for " + model_short_name
          + " (seed " + str(seed) + "):")
    print("  weighted F1 = %.4f   macro F1 = %.4f   accuracy = %.4f" % (
        metrics["weighted_f1"], metrics["macro_f1"], metrics["accuracy"]))
    print("  saved to " + run_dir)

    return metrics
