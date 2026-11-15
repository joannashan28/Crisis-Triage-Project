"""
Metrics shared by every model and every stage, so all results are computed
exactly the same way.

Primary metric: weighted F1 (F1 per class, averaged with each class weighted
by how many test tweets it has). Secondary: macro F1 (every class counts
equally, so rare classes like missing/found people matter as much as common
ones), plus precision, recall, and accuracy.
"""

import matplotlib

# "Agg" draws plots straight to image files, so this works on servers and
# in Colab without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config


def compute_metrics(gold_labels, predicted_labels, labels_in_order):
    """Compute overall, per-class, and confusion-matrix results."""
    results = {}
    results["n"] = len(gold_labels)
    results["accuracy"] = accuracy_score(gold_labels, predicted_labels)

    for average in ["weighted", "macro"]:
        precision, recall, f1, support = precision_recall_fscore_support(
            gold_labels,
            predicted_labels,
            labels=labels_in_order,
            average=average,
            zero_division=0,
        )
        results[average + "_precision"] = float(precision)
        results[average + "_recall"] = float(recall)
        results[average + "_f1"] = float(f1)

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        gold_labels,
        predicted_labels,
        labels=labels_in_order,
        average=None,
        zero_division=0,
    )
    per_class = {}
    for index in range(len(labels_in_order)):
        label = labels_in_order[index]
        per_class[label] = {
            "precision": float(precisions[index]),
            "recall": float(recalls[index]),
            "f1": float(f1s[index]),
            "support": int(supports[index]),
        }
    results["per_class"] = per_class

    matrix = confusion_matrix(gold_labels, predicted_labels, labels=labels_in_order)
    results["confusion_matrix"] = {
        "labels": list(labels_in_order),
        "rows_are_gold_columns_are_predicted": matrix.tolist(),
    }

    return results


def metrics_by_group(rows, gold_labels, predicted_labels, group_column):
    """Compute weighted F1, macro F1, and accuracy separately for each value
    of a column (for example disaster_type or source).

    Within a group, macro F1 only averages over labels that actually appear
    in that group (as gold or predicted). Otherwise a group with no
    "missing people" tweets would be penalized for that class."""
    gold_by_group = {}
    predicted_by_group = {}

    for index in range(len(rows)):
        group = rows[index][group_column]
        if group not in gold_by_group:
            gold_by_group[group] = []
            predicted_by_group[group] = []
        gold_by_group[group].append(gold_labels[index])
        predicted_by_group[group].append(predicted_labels[index])

    results = {}
    for group in sorted(gold_by_group):
        group_gold = gold_by_group[group]
        group_predicted = predicted_by_group[group]

        labels_present = set()
        labels_present.update(group_gold)
        labels_present.update(group_predicted)
        labels_present = sorted(labels_present)

        group_results = {"n": len(group_gold)}
        group_results["accuracy"] = accuracy_score(group_gold, group_predicted)
        for average in ["weighted", "macro"]:
            precision, recall, f1, support = precision_recall_fscore_support(
                group_gold,
                group_predicted,
                labels=labels_present,
                average=average,
                zero_division=0,
            )
            group_results[average + "_f1"] = float(f1)
        results[group] = group_results

    return results


def save_confusion_matrix_plot(metrics, path, title):
    """Save a confusion matrix image. Each row is normalized to sum to 1, so
    the diagonal shows recall for each gold class, and the colors are
    readable even when classes have very different sizes. The raw counts
    are printed in each cell."""
    labels = metrics["confusion_matrix"]["labels"]
    counts = metrics["confusion_matrix"]["rows_are_gold_columns_are_predicted"]

    normalized = []
    for row in counts:
        row_total = sum(row)
        normalized_row = []
        for value in row:
            if row_total == 0:
                normalized_row.append(0.0)
            else:
                normalized_row.append(value / row_total)
        normalized.append(normalized_row)

    display_names = []
    for label in labels:
        entry = config.get_label_display(label)
        display_names.append(entry["display"])

    size = max(5, len(labels) * 0.9)
    figure, axes = plt.subplots(figsize=(size + 2, size))
    image = axes.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04,
                    label="Share of the gold class")

    axes.set_xticks(range(len(labels)))
    axes.set_yticks(range(len(labels)))
    axes.set_xticklabels(display_names, rotation=45, ha="right")
    axes.set_yticklabels(display_names)
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("Gold label")
    axes.set_title(title)

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            share = normalized[row_index][column_index]
            if share > 0.5:
                text_color = "white"
            else:
                text_color = "black"
            axes.text(column_index, row_index, str(counts[row_index][column_index]),
                      ha="center", va="center", fontsize=8, color=text_color)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
