"""
Baselines for both stages.

1. Majority class: always predict the most common training label. This is
   the floor every real model has to beat. On Stage 2 it gets a decent
   weighted F1 but a terrible macro F1, which is exactly why we report both.

2. TF-IDF + logistic regression: a bag-of-words model (single words and
   word pairs). We pick the regularization strength C using the dev set.
   For Stage 2 we use the same sqrt-balanced class weights as BERTweet, so
   the comparison between the two is fair.

Both are deterministic, so they are saved as seed 0 and only need to run once.
The TF-IDF models are also saved so the demo server can use them as a backup
if the BERTweet models are not ready.

Usage:
    python src/run_baselines.py
    python src/run_baselines.py --stages 2      # only Stage 2
"""

import argparse
import os
import time
from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline

import config
from class_weights import WEIGHTING_CHOICES, compute_class_weights
from crisisbench_io import clean_for_tfidf, load_processed_split, save_json
from run_outputs import run_directory, save_run


C_VALUES_TO_TRY = [0.25, 1.0, 4.0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=config.PROCESSED_DATA_DIR)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--stage2-class-weighting", default="sqrt_balanced",
                        choices=WEIGHTING_CHOICES)
    return parser.parse_args()


def get_texts_and_labels(rows):
    texts = []
    labels = []
    for row in rows:
        texts.append(row["text"])
        labels.append(row["label"])
    return texts, labels


def sorted_labels(labels):
    unique = set(labels)
    return sorted(unique)


def informative_probabilities_from(probability_rows, class_order):
    """Pull out P(informative) from a predict_proba result."""
    informative_index = class_order.index(config.INFORMATIVE_LABEL)
    probabilities = []
    for probability_row in probability_rows:
        probabilities.append(float(probability_row[informative_index]))
    return probabilities


# ---------------------------------------------------------------------------
# Majority class
# ---------------------------------------------------------------------------

def run_majority(stage, train_rows, test_rows):
    train_texts, train_labels = get_texts_and_labels(train_rows)
    labels_in_order = sorted_labels(train_labels)

    counts = Counter(train_labels)
    most_common_label, most_common_count = counts.most_common(1)[0]
    share = most_common_count / len(train_labels)

    predicted = []
    confidences = []
    for row in test_rows:
        predicted.append(most_common_label)
        confidences.append(share)

    informative_probabilities = None
    if stage == 1:
        # The baseline's "probability" of informative is just how often
        # informative appears in training.
        informative_share = counts[config.INFORMATIVE_LABEL] / len(train_labels)
        informative_probabilities = []
        for row in test_rows:
            informative_probabilities.append(informative_share)

    settings = {"always_predicts": most_common_label}
    run_dir = run_directory("majority", stage, 0)
    save_run(run_dir, "majority", stage, 0, test_rows, predicted, confidences,
             informative_probabilities, labels_in_order, settings)


# ---------------------------------------------------------------------------
# TF-IDF + logistic regression
# ---------------------------------------------------------------------------

def build_tfidf_model(c_value, class_weight):
    vectorizer = TfidfVectorizer(
        preprocessor=clean_for_tfidf,
        ngram_range=(1, 2),
        min_df=2,
        max_features=200000,
        sublinear_tf=True,
    )
    classifier = LogisticRegression(
        C=c_value,
        max_iter=2000,
        class_weight=class_weight,
    )
    model = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
    return model


def run_tfidf(stage, train_rows, dev_rows, test_rows, weighting):
    train_texts, train_labels = get_texts_and_labels(train_rows)
    dev_texts, dev_labels = get_texts_and_labels(dev_rows)
    test_texts, test_labels = get_texts_and_labels(test_rows)
    labels_in_order = sorted_labels(train_labels)

    weights = compute_class_weights(labels_in_order, train_labels, weighting)
    class_weight = {}
    for index in range(len(labels_in_order)):
        class_weight[labels_in_order[index]] = weights[index]

    # Choose C on the dev set, using weighted F1 (our primary metric).
    best_model = None
    best_c = None
    best_dev_f1 = -1.0
    dev_scores = {}
    for c_value in C_VALUES_TO_TRY:
        start = time.time()
        model = build_tfidf_model(c_value, class_weight)
        model.fit(train_texts, train_labels)
        dev_predictions = model.predict(dev_texts)
        precision, recall, dev_f1, support = precision_recall_fscore_support(
            dev_labels, dev_predictions, labels=labels_in_order,
            average="weighted", zero_division=0)
        dev_scores[str(c_value)] = float(dev_f1)
        elapsed = time.time() - start
        print("  Stage %d, C=%s: dev weighted F1 = %.4f (%.0fs)" % (
            stage, c_value, dev_f1, elapsed))

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_c = c_value
            best_model = model

    # Predict on test with the chosen model.
    probability_rows = best_model.predict_proba(test_texts)
    class_order = list(best_model.classes_)

    predicted = []
    confidences = []
    for probability_row in probability_rows:
        best_index = 0
        for index in range(len(probability_row)):
            if probability_row[index] > probability_row[best_index]:
                best_index = index
        predicted.append(class_order[best_index])
        confidences.append(float(probability_row[best_index]))

    informative_probabilities = None
    if stage == 1:
        informative_probabilities = informative_probabilities_from(probability_rows, class_order)

    settings = {
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_features": 200000,
        "chosen_C": best_c,
        "dev_weighted_f1_by_C": dev_scores,
        "class_weighting": weighting,
        "class_weights": class_weight,
        "train_size": len(train_rows),
    }
    run_dir = run_directory("tfidf_lr", stage, 0)
    save_run(run_dir, "tfidf_lr", stage, 0, test_rows, predicted, confidences,
             informative_probabilities, labels_in_order, settings)

    # Save the model for the demo server.
    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "model.joblib"))
    description = {
        "kind": "tfidf",
        "stage": stage,
        "labels": class_order,
        "model_file": "model.joblib",
    }
    save_json(description, os.path.join(model_dir, "triage_model.json"))


def main():
    args = parse_args()

    for stage in args.stages:
        print("=== Stage " + str(stage) + " baselines ===")
        train_rows = load_processed_split(args.data_dir, stage, "train")
        dev_rows = load_processed_split(args.data_dir, stage, "dev")
        test_rows = load_processed_split(args.data_dir, stage, "test")

        run_majority(stage, train_rows, test_rows)

        if stage == 1:
            weighting = "none"
        else:
            weighting = args.stage2_class_weighting
        run_tfidf(stage, train_rows, dev_rows, test_rows, weighting)


if __name__ == "__main__":
    main()
