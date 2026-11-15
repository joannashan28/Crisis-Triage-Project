"""
Collect every metrics.json under runs/ and average across seeds.

Writes:
  results/summary.md       tables ready to paste into the report (mean +/- std)
  results/summary.csv      the same numbers, one row per model and stage
  results/per_class_f1.csv Stage 2 and end-to-end F1 for each category

Standard deviation is across seeds (sample std, n - 1). A model with one
seed shows no std. Deterministic baselines (majority, TF-IDF) only ever
have seed 0.

Usage:
    python src/summarize_results.py
"""

import argparse
import csv
import math
import os

import config
from crisisbench_io import load_json


SUMMARY_METRICS = [
    "weighted_f1",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "accuracy",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default=config.RUNS_DIR)
    parser.add_argument("--output-dir", default=config.RESULTS_DIR)
    return parser.parse_args()


def find_metrics_files(runs_dir):
    paths = []
    for folder, subfolders, files in os.walk(runs_dir):
        if "metrics.json" in files:
            paths.append(os.path.join(folder, "metrics.json"))
    return sorted(paths)


def mean_and_std(values):
    count = len(values)
    mean = sum(values) / count
    if count < 2:
        return mean, None
    squared_differences = 0.0
    for value in values:
        squared_differences += (value - mean) ** 2
    std = math.sqrt(squared_differences / (count - 1))
    return mean, std


def format_mean_std(mean, std):
    if std is None:
        return "%.4f" % mean
    return "%.4f ± %.4f" % (mean, std)


def group_runs(metrics_list):
    """Group runs that differ only by seed. Returns an ordered dict-like
    list of (group_name, section, list_of_metrics)."""
    groups = {}
    order = []
    for metrics in metrics_list:
        if "is_end_to_end" in metrics:
            section = "end_to_end"
            name = metrics["system"]
            if metrics["threshold"] != config.DEFAULT_STAGE1_THRESHOLD:
                name = name + " (threshold " + str(metrics["threshold"]) + ")"
        else:
            section = "stage" + str(metrics["stage"])
            name = metrics["model"]

        key = (section, name)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(metrics)

    result = []
    for key in order:
        section, name = key
        result.append((section, name, groups[key]))
    return result


def summarize_group(runs):
    summary = {"seeds": len(runs)}
    for metric in SUMMARY_METRICS:
        values = []
        for run in runs:
            values.append(run[metric])
        mean, std = mean_and_std(values)
        summary[metric] = (mean, std)
    return summary


def per_class_means(runs):
    """Average each class's F1 across seeds."""
    values_by_label = {}
    for run in runs:
        for label in run["per_class"]:
            if label not in values_by_label:
                values_by_label[label] = []
            values_by_label[label].append(run["per_class"][label]["f1"])

    means = {}
    for label in values_by_label:
        mean, std = mean_and_std(values_by_label[label])
        means[label] = mean
    return means


def markdown_table(section_title, rows):
    lines = []
    lines.append("## " + section_title)
    lines.append("")
    header = "| Model | Seeds | Weighted F1 | Macro F1 | Weighted P | Weighted R | Accuracy |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|")
    for name, summary in rows:
        cells = [name, str(summary["seeds"])]
        for metric in SUMMARY_METRICS:
            mean, std = summary[metric]
            cells.append(format_mean_std(mean, std))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_list = []
    for path in find_metrics_files(args.runs_dir):
        metrics_list.append(load_json(path))

    if len(metrics_list) == 0:
        print("No metrics.json files found under " + args.runs_dir)
        return

    grouped = group_runs(metrics_list)

    section_titles = {
        "stage1": "Stage 1: informative vs. not informative (official test set)",
        "stage2": "Stage 2: humanitarian category (official test set)",
        "end_to_end": "End to end: full pipeline (tweets in both test sets)",
    }

    markdown_lines = ["# Results summary", "",
                      "Mean ± standard deviation across seeds.", ""]
    csv_rows = []
    per_class_rows = []

    for section in ["stage1", "stage2", "end_to_end"]:
        table_rows = []
        for group_section, name, runs in grouped:
            if group_section != section:
                continue

            summary = summarize_group(runs)
            table_rows.append((name, summary))

            csv_row = {"section": section, "model": name, "seeds": summary["seeds"]}
            for metric in SUMMARY_METRICS:
                mean, std = summary[metric]
                csv_row[metric + "_mean"] = "%.4f" % mean
                if std is None:
                    csv_row[metric + "_std"] = ""
                else:
                    csv_row[metric + "_std"] = "%.4f" % std
            csv_rows.append(csv_row)

            if section != "stage1":
                means = per_class_means(runs)
                for label in sorted(means):
                    per_class_rows.append({
                        "section": section,
                        "model": name,
                        "label": label,
                        "display_name": config.get_label_display(label)["display"],
                        "f1_mean": "%.4f" % means[label],
                    })

        if len(table_rows) > 0:
            markdown_lines.extend(markdown_table(section_titles[section], table_rows))

    markdown_path = os.path.join(args.output_dir, "summary.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines))

    csv_columns = ["section", "model", "seeds"]
    for metric in SUMMARY_METRICS:
        csv_columns.append(metric + "_mean")
        csv_columns.append(metric + "_std")
    write_csv(os.path.join(args.output_dir, "summary.csv"), csv_rows, csv_columns)
    write_csv(os.path.join(args.output_dir, "per_class_f1.csv"), per_class_rows,
              ["section", "model", "label", "display_name", "f1_mean"])

    print("\n".join(markdown_lines))
    print("Saved tables to " + args.output_dir)


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
