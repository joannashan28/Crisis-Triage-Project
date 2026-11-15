"""
Manual error analysis, in two steps.

Step 1 (export): pull a random sample of misclassified tweets from an
end-to-end run, a fixed number per error type, into a spreadsheet with an
empty cause_tag column for you to fill in.

    python src/error_review.py export \
        --run runs/end_to_end/stage1-bertweet-base-seed1__stage2-bertweet-base-seed1 \
        --per-type 30

Error types (set by evaluate_end_to_end.py):
    stage1_filtered_actionable  a real report thrown out as noise by Stage 1
    stage2_marked_other         passed Stage 1, but Stage 2 said "not humanitarian"
    stage2_wrong_category       passed Stage 1, but Stage 2 picked the wrong category
    noise_passed_through        not actionable, but ended up in a category

Suggested cause tags (from our proposal; add more if you need them):
    ambiguous_or_noisy, sarcasm_or_missing_context, overlapping_categories,
    label_looks_wrong, stage1_error, other

Step 2 (summarize): after filling in cause_tag, count tags per error type.

    python src/error_review.py summarize --file results/error_review.csv
"""

import argparse
import csv
import os
import random
from collections import Counter

import config
from crisisbench_io import read_tsv, save_json


EXPORT_COLUMNS = [
    "id",
    "error_type",
    "gold_end_to_end",
    "pred_end_to_end",
    "gold_informativeness",
    "gold_humanitarian",
    "stage1_prob_informative",
    "stage2_pred",
    "disaster_type",
    "source",
    "text",
    "cause_tag",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--run", required=True, help="An end-to-end run folder")
    export_parser.add_argument("--per-type", type=int, default=30)
    export_parser.add_argument("--seed", type=int, default=0)
    export_parser.add_argument("--output",
                               default=os.path.join(config.RESULTS_DIR, "error_review.csv"))

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--file", required=True)

    return parser.parse_args()


def export(args):
    rows = read_tsv(os.path.join(args.run, "predictions.tsv"))

    rows_by_type = {}
    for row in rows:
        error_type = row["error_type"]
        if error_type == "correct":
            continue
        if error_type not in rows_by_type:
            rows_by_type[error_type] = []
        rows_by_type[error_type].append(row)

    sampler = random.Random(args.seed)
    selected = []
    for error_type in sorted(rows_by_type):
        candidates = rows_by_type[error_type]
        if len(candidates) <= args.per_type:
            chosen = candidates
        else:
            chosen = sampler.sample(candidates, args.per_type)
        print("%-28s %5d errors total, %3d exported" % (
            error_type, len(candidates), len(chosen)))
        selected.extend(chosen)

    folder = os.path.dirname(args.output)
    os.makedirs(folder, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            output_row = dict(row)
            output_row["cause_tag"] = ""
            output_row["notes"] = ""
            writer.writerow(output_row)

    print("Wrote " + str(len(selected)) + " tweets to " + args.output)
    print("Fill in the cause_tag column, then run: "
          "python src/error_review.py summarize --file " + args.output)


def summarize(args):
    with open(args.file, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(row)

    counts_by_type = {}
    untagged = 0
    for row in rows:
        tag = row["cause_tag"].strip()
        if tag == "":
            untagged += 1
            continue
        error_type = row["error_type"]
        if error_type not in counts_by_type:
            counts_by_type[error_type] = Counter()
        counts_by_type[error_type][tag] += 1

    summary = {}
    for error_type in sorted(counts_by_type):
        print(error_type)
        counts = counts_by_type[error_type]
        total = sum(counts.values())
        summary[error_type] = {}
        for tag, count in counts.most_common():
            share = count / total
            print("    %-30s %4d  (%.0f%%)" % (tag, count, 100 * share))
            summary[error_type][tag] = count

    if untagged > 0:
        print(str(untagged) + " rows have no cause_tag yet.")

    output_path = os.path.splitext(args.file)[0] + "_summary.json"
    save_json(summary, output_path)
    print("Saved counts to " + output_path)


def main():
    args = parse_args()
    if args.command == "export":
        export(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
