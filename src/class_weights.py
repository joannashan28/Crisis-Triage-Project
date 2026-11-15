"""
Class weights for training on imbalanced data.

Stage 2 is very imbalanced: "not humanitarian" has about 100 times as many
training tweets as "missing and found people". Without weights, the model
can score well overall while mostly ignoring the rare classes, which are the
ones responders care about most.

Three options:
  none           every class counts the same (weight 1.0)
  balanced       weight = total / (number_of_classes * class_count).
                 A class with 100x fewer examples gets 100x the weight.
  sqrt_balanced  the square root of "balanced". A class with 100x fewer
                 examples gets 10x the weight. This still boosts rare
                 classes, but less aggressively, which usually keeps
                 training more stable and avoids flooding the rare
                 categories with false positives.

We use sqrt_balanced for Stage 2 by default and none for Stage 1 (which is
only mildly imbalanced). Both are command-line options, so you can compare.
"""

import math
from collections import Counter


WEIGHTING_CHOICES = ["none", "balanced", "sqrt_balanced"]


def compute_class_weights(labels_in_order, training_labels, scheme):
    """Return a list of weights, one per label, in the same order as
    labels_in_order."""
    if scheme not in WEIGHTING_CHOICES:
        raise ValueError("Unknown class weighting: " + scheme)

    counts = Counter(training_labels)
    total = len(training_labels)
    number_of_classes = len(labels_in_order)

    weights = []
    for label in labels_in_order:
        if scheme == "none":
            weights.append(1.0)
            continue

        count = counts[label]
        if count == 0:
            # A label with no training examples cannot be learned anyway.
            weights.append(1.0)
            continue

        balanced_weight = total / (number_of_classes * count)
        if scheme == "balanced":
            weights.append(balanced_weight)
        else:
            weights.append(math.sqrt(balanced_weight))

    return weights
