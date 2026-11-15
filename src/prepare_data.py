"""
Build the data files every other script uses.

What this script does, in order:

1. Reads CrisisBench's official train/dev/test files for both tasks:
     - informativeness  -> Stage 1 (informative vs. not informative)
     - humanitarian     -> Stage 2 (which humanitarian category)
2. Works out each tweet's disaster type (flood, earthquake, ...) from its
   event name, so results can be broken down by disaster type later.
   The keyword rules are in config.py.
3. Removes leakage. The two tasks were split separately, so a tweet in one
   task's TEST set can show up in the other task's TRAIN set. Because the
   pipeline uses both stages together, that would let the model see test
   tweets during training. We remove from every training set any tweet whose
   ID or text also appears in any dev or test set, and from every dev set
   any tweet that also appears in any test set. Test sets are never changed,
   so our test numbers stay comparable to published CrisisBench results.
4. Builds the end-to-end test set: tweets that appear in BOTH test sets, so
   they have a gold label for each stage. This is also the pool of tweets
   shown in the demo feed.
5. Writes everything to data/processed/ along with data_report.json, which
   records every count (including how many tweets the leakage step removed).

Usage:
    python src/prepare_data.py --crisisbench-dir /path/to/crisis_datasets_benchmarks
"""

import argparse
import os
from collections import Counter

import config
from crisisbench_io import (
    find_file,
    length_bucket,
    normalize_for_matching,
    read_tsv,
    read_tsv_columns,
    save_json,
    write_tsv,
)


REQUIRED_COLUMNS = ["id", "event", "source", "text", "class_label"]

OUTPUT_COLUMNS = ["id", "text", "label", "event", "source", "disaster_type"]

END_TO_END_COLUMNS = [
    "id",
    "text",
    "gold_informativeness",
    "gold_humanitarian",
    "gold_end_to_end",
    "event",
    "source",
    "disaster_type",
    "length_bucket",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crisisbench-dir", required=True,
                        help="Folder where you extracted crisis_datasets_benchmarks_v1.0.tar.gz")
    parser.add_argument("--output-dir", default=config.PROCESSED_DATA_DIR)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_task_split(crisisbench_dir, template, split):
    """Read one official split file (e.g. humanitarian train) and check that
    it has the columns we expect."""
    file_name = template.format(split=split)
    path = find_file(crisisbench_dir, file_name)
    if path is None:
        raise FileNotFoundError(
            "Could not find " + file_name + " anywhere inside " + crisisbench_dir
        )

    columns = read_tsv_columns(path)
    for column in REQUIRED_COLUMNS:
        if column not in columns:
            raise RuntimeError(
                path + " is missing the column '" + column + "'. "
                "Its columns are: " + ", ".join(columns)
            )

    print("Reading " + path)
    rows = read_tsv(path)
    return rows


def remove_duplicate_ids(rows):
    """Keep only the first row for each tweet ID. Returns the kept rows and
    how many were dropped."""
    seen_ids = set()
    kept_rows = []
    dropped = 0
    for row in rows:
        tweet_id = row["id"]
        if tweet_id in seen_ids:
            dropped = dropped + 1
            continue
        seen_ids.add(tweet_id)
        kept_rows.append(row)
    return kept_rows, dropped


def to_output_rows(raw_rows):
    """Convert CrisisBench rows into the smaller format our scripts use."""
    output_rows = []
    for raw in raw_rows:
        disaster_type = config.disaster_type_from_event(raw["event"])

        output_row = {
            "id": raw["id"],
            "text": raw["text"].strip(),
            "label": raw["class_label"].strip(),
            "event": raw["event"],
            "source": raw["source"],
            "disaster_type": disaster_type,
        }
        output_rows.append(output_row)
    return output_rows


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_labels(stage1_rows, stage2_rows):
    """Make sure the label strings in config.py match the data. If they do
    not, stop with a message showing the labels that are really there."""
    stage1_labels = set()
    for row in stage1_rows:
        stage1_labels.add(row["label"])

    stage2_labels = set()
    for row in stage2_rows:
        stage2_labels.add(row["label"])

    problems = []
    if config.INFORMATIVE_LABEL not in stage1_labels:
        problems.append("INFORMATIVE_LABEL '" + config.INFORMATIVE_LABEL + "'")
    if config.NOT_INFORMATIVE_LABEL not in stage1_labels:
        problems.append("NOT_INFORMATIVE_LABEL '" + config.NOT_INFORMATIVE_LABEL + "'")
    if config.NOT_HUMANITARIAN_LABEL not in stage2_labels:
        problems.append("NOT_HUMANITARIAN_LABEL '" + config.NOT_HUMANITARIAN_LABEL + "'")

    if len(problems) > 0:
        message = "These labels in config.py were not found in the data: "
        message = message + ", ".join(problems) + "\n"
        message = message + "Stage 1 labels in the data: " + ", ".join(sorted(stage1_labels)) + "\n"
        message = message + "Stage 2 labels in the data: " + ", ".join(sorted(stage2_labels))
        raise RuntimeError(message)


def collect_ids_and_texts(row_lists):
    """Gather every tweet ID and every normalized tweet text from several
    lists of rows. Used to decide what counts as leakage."""
    ids = set()
    texts = set()
    for rows in row_lists:
        for row in rows:
            ids.add(row["id"])
            normalized = normalize_for_matching(row["text"])
            if normalized != "":
                texts.add(normalized)
    return ids, texts


def remove_overlap(rows, blocked_ids, blocked_texts):
    """Drop rows whose ID or normalized text is in the blocked sets.
    Returns the kept rows and counts of what was removed and why."""
    kept_rows = []
    removed_by_id = 0
    removed_by_text = 0

    for row in rows:
        if row["id"] in blocked_ids:
            removed_by_id = removed_by_id + 1
            continue

        normalized = normalize_for_matching(row["text"])
        if normalized in blocked_texts:
            removed_by_text = removed_by_text + 1
            continue

        kept_rows.append(row)

    counts = {"removed_same_id": removed_by_id, "removed_same_text": removed_by_text}
    return kept_rows, counts


# ---------------------------------------------------------------------------
# End-to-end test set
# ---------------------------------------------------------------------------

def gold_end_to_end_label(informativeness_label, humanitarian_label):
    """The single gold label we score the whole pipeline against.

    We follow the same order as the pipeline: Stage 1's label is checked
    first. If annotators said the tweet is not informative, the correct
    pipeline output is 'not actionable', whatever its humanitarian label.
    Otherwise the humanitarian label is the answer, with 'not humanitarian'
    also counted as 'not actionable'.
    """
    if informativeness_label == config.NOT_INFORMATIVE_LABEL:
        return config.NOT_ACTIONABLE_LABEL
    if humanitarian_label == config.NOT_HUMANITARIAN_LABEL:
        return config.NOT_ACTIONABLE_LABEL
    return humanitarian_label


def build_end_to_end_set(stage1_test, stage2_test):
    """Join the two test sets on tweet ID. Also count tweets where the two
    sets of annotations disagree, so we can report it."""
    stage2_by_id = {}
    for row in stage2_test:
        stage2_by_id[row["id"]] = row

    end_to_end_rows = []
    conflicts = Counter()

    for stage1_row in stage1_test:
        tweet_id = stage1_row["id"]
        if tweet_id not in stage2_by_id:
            continue

        stage2_row = stage2_by_id[tweet_id]
        info_label = stage1_row["label"]
        hum_label = stage2_row["label"]

        # Annotation conflicts: "not informative" but given a real
        # humanitarian category, or "informative" but "not humanitarian".
        # The second is not really a conflict (an informative tweet can fall
        # outside the categories), but it is useful to know how common it is.
        if info_label == config.NOT_INFORMATIVE_LABEL:
            if hum_label != config.NOT_HUMANITARIAN_LABEL:
                conflicts["not_informative_but_has_category"] += 1
        else:
            if hum_label == config.NOT_HUMANITARIAN_LABEL:
                conflicts["informative_but_not_humanitarian"] += 1

        row = {
            "id": tweet_id,
            "text": stage1_row["text"],
            "gold_informativeness": info_label,
            "gold_humanitarian": hum_label,
            "gold_end_to_end": gold_end_to_end_label(info_label, hum_label),
            "event": stage1_row["event"],
            "source": stage1_row["source"],
            "disaster_type": stage1_row["disaster_type"],
            "length_bucket": length_bucket(stage1_row["text"]),
        }
        end_to_end_rows.append(row)

    return end_to_end_rows, dict(conflicts)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_disaster_types(data, report):
    """Record which disaster type each event name was given, and print any
    event names that matched no rule in config.py."""
    event_counts = Counter()
    for stage_name in data:
        for split in config.SPLITS:
            for row in data[stage_name][split]:
                event_counts[row["event"]] += 1

    event_to_type = {}
    unmatched = {}
    for event, count in event_counts.most_common():
        disaster_type = config.disaster_type_from_event(event)
        event_to_type[event] = disaster_type
        if disaster_type == config.UNKNOWN_DISASTER_TYPE:
            unmatched[event] = count

    report["event_to_disaster_type"] = event_to_type
    report["events_with_no_disaster_type"] = unmatched

    print("Gave a disaster type to " + str(len(event_counts) - len(unmatched))
          + " of " + str(len(event_counts)) + " event names.")
    if len(unmatched) > 0:
        print("These event names matched no rule and are tagged 'unknown'.")
        print("If any clearly has a type, add a keyword to DISASTER_TYPE_RULES in config.py:")
        for event in unmatched:
            print("    %6d tweets  %s" % (unmatched[event], event))


def count_labels(rows, label_column):
    counts = Counter()
    for row in rows:
        counts[row[label_column]] += 1
    return dict(counts.most_common())


def count_disaster_types(rows):
    counts = Counter()
    for row in rows:
        counts[row["disaster_type"]] += 1
    return dict(counts.most_common())


def main():
    args = parse_args()
    report = {}

    # 1. Load both tasks ----------------------------------------------------
    stages = {
        "stage1": config.INFORMATIVENESS_FILE_TEMPLATE,
        "stage2": config.HUMANITARIAN_FILE_TEMPLATE,
    }

    data = {}
    report["duplicate_ids_dropped"] = {}
    for stage_name in stages:
        template = stages[stage_name]
        data[stage_name] = {}
        for split in config.SPLITS:
            raw_rows = load_task_split(args.crisisbench_dir, template, split)
            raw_rows, dropped = remove_duplicate_ids(raw_rows)
            report["duplicate_ids_dropped"][stage_name + "_" + split] = dropped
            data[stage_name][split] = to_output_rows(raw_rows)

    check_labels(data["stage1"]["train"], data["stage2"]["train"])
    report_disaster_types(data, report)

    report["original_sizes"] = {}
    for stage_name in data:
        for split in config.SPLITS:
            key = stage_name + "_" + split
            report["original_sizes"][key] = len(data[stage_name][split])

    # 2. Remove leakage -----------------------------------------------------
    # Anything in either task's test set is off limits for training and dev.
    test_ids, test_texts = collect_ids_and_texts([
        data["stage1"]["test"],
        data["stage2"]["test"],
    ])
    # Anything in either task's dev set is also off limits for training,
    # so model selection on dev is not affected by memorized tweets.
    dev_ids, dev_texts = collect_ids_and_texts([
        data["stage1"]["dev"],
        data["stage2"]["dev"],
    ])

    held_out_ids = set()
    held_out_ids.update(test_ids)
    held_out_ids.update(dev_ids)
    held_out_texts = set()
    held_out_texts.update(test_texts)
    held_out_texts.update(dev_texts)

    report["leakage_removed"] = {}
    for stage_name in data:
        train_rows, train_counts = remove_overlap(
            data[stage_name]["train"], held_out_ids, held_out_texts)
        dev_rows, dev_counts = remove_overlap(
            data[stage_name]["dev"], test_ids, test_texts)

        data[stage_name]["train"] = train_rows
        data[stage_name]["dev"] = dev_rows
        report["leakage_removed"][stage_name + "_train"] = train_counts
        report["leakage_removed"][stage_name + "_dev"] = dev_counts

    # 3. Write per-stage files ----------------------------------------------
    report["final_sizes"] = {}
    report["label_counts"] = {}
    for stage_name in data:
        for split in config.SPLITS:
            rows = data[stage_name][split]
            key = stage_name + "_" + split
            path = os.path.join(args.output_dir, key + ".tsv")
            write_tsv(path, rows, OUTPUT_COLUMNS)
            report["final_sizes"][key] = len(rows)
            report["label_counts"][key] = count_labels(rows, "label")
            print("Wrote " + str(len(rows)) + " rows to " + path)

    # 4. End-to-end test set ------------------------------------------------
    end_to_end_rows, conflicts = build_end_to_end_set(
        data["stage1"]["test"], data["stage2"]["test"])
    path = os.path.join(args.output_dir, "end_to_end_test.tsv")
    write_tsv(path, end_to_end_rows, END_TO_END_COLUMNS)
    print("Wrote " + str(len(end_to_end_rows)) + " rows to " + path)

    report["end_to_end_test_size"] = len(end_to_end_rows)
    report["end_to_end_label_counts"] = count_labels(end_to_end_rows, "gold_end_to_end")
    report["end_to_end_annotation_conflicts"] = conflicts
    report["end_to_end_disaster_types"] = count_disaster_types(end_to_end_rows)
    report["stage1_test_disaster_types"] = count_disaster_types(data["stage1"]["test"])
    report["stage2_test_disaster_types"] = count_disaster_types(data["stage2"]["test"])

    report_path = os.path.join(args.output_dir, "data_report.json")
    save_json(report, report_path)
    print("Saved counts and leakage report to " + report_path)


if __name__ == "__main__":
    main()
