"""
Small helpers for reading and writing data files and for cleaning tweet text.

We use Python's built-in csv module instead of pandas so every step of
reading and writing is visible and easy to explain.
"""

import csv
import json
import os
import re
import sys


# Some tweets are long enough to trip csv's default field-size limit.
csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------------------
# Finding and reading CrisisBench files
# ---------------------------------------------------------------------------

def find_file(root_dir, file_name):
    """Search root_dir (and every folder inside it) for a file with this exact
    name. Returns the full path, or None if the file is not there.

    If the same file name appears in more than one folder, we prefer the copy
    in a folder called all_data_en, since that is the combined English data
    CrisisBench uses for its benchmark results.
    """
    matches = []
    for folder, subfolders, files in os.walk(root_dir):
        if file_name in files:
            full_path = os.path.join(folder, file_name)
            matches.append(full_path)

    if len(matches) == 0:
        return None

    if len(matches) == 1:
        return matches[0]

    for path in matches:
        folder_name = os.path.basename(os.path.dirname(path))
        if folder_name == "all_data_en":
            return path

    message = "Found more than one copy of " + file_name + ":\n"
    for path in matches:
        message = message + "  " + path + "\n"
    message = message + "Point --crisisbench-dir at the folder that holds the one you want."
    raise RuntimeError(message)


def read_tsv(path):
    """Read a tab-separated file into a list of dictionaries (one per row)."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            rows.append(row)
    return rows


def read_tsv_columns(path):
    """Return just the column names (header row) of a tab-separated file."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
    return header


def write_tsv(path, rows, columns):
    """Write a list of dictionaries to a tab-separated file.

    Only the listed columns are written, in that order. Tabs and newlines
    inside values are replaced with spaces so each row stays on one line.
    """
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        # quotechar=None means quotes inside tweets are written exactly as
        # they are, which matches how read_tsv reads them back.
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_NONE,
                            quotechar=None, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                value = str(value)
                value = value.replace("\t", " ")
                value = value.replace("\n", " ")
                value = value.replace("\r", " ")
                values.append(value)
            writer.writerow(values)


def load_processed_split(data_dir, stage, split):
    """Read one of the files written by prepare_data.py, e.g. stage2_train.tsv."""
    file_name = "stage" + str(stage) + "_" + split + ".tsv"
    path = os.path.join(data_dir, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(path + " does not exist. Run src/prepare_data.py first.")
    return read_tsv(path)


def save_json(data, path):
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
NON_WORD_PATTERN = re.compile(r"[^a-z0-9#\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_for_matching(text):
    """Reduce a tweet to a simple form used ONLY for spotting duplicates
    between files (the leakage check). Two tweets that differ only in
    their links, @mentions, a leading "RT", punctuation, or capitalization
    count as the same tweet.
    """
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = NON_WORD_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = text.strip()

    if text.startswith("rt "):
        text = text[3:]

    return text


def clean_for_tfidf(text):
    """Light cleanup for the TF-IDF baseline: lowercase, and replace links and
    @mentions with placeholder words. This mirrors what BERTweet's own
    normalization does (it uses HTTPURL and @USER), so the baseline and
    BERTweet see roughly the same information.

    This function must stay importable from this module, because the saved
    TF-IDF model refers to it by name.
    """
    text = text.lower()
    text = URL_PATTERN.sub(" httpurl ", text)
    text = MENTION_PATTERN.sub(" @user ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = text.strip()
    return text


def word_count(text):
    words = text.split()
    return len(words)


def length_bucket(text):
    """Group tweets by length for the error breakdown."""
    count = word_count(text)
    if count <= 10:
        return "short (1-10 words)"
    if count <= 20:
        return "medium (11-20 words)"
    return "long (21+ words)"
