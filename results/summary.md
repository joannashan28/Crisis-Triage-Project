# Results summary

Mean ± standard deviation across seeds.

## Stage 1: informative vs. not informative (official test set)

| Model | Seeds | Weighted F1 | Macro F1 | Weighted P | Weighted R | Accuracy |
|---|---|---|---|---|---|---|
| bertweet-base | 1 | 0.8861 | 0.8805 | 0.8876 | 0.8871 | 0.8871 |
| majority | 1 | 0.4488 | 0.3746 | 0.3588 | 0.5990 | 0.5990 |
| tfidf_lr | 1 | 0.8611 | 0.8546 | 0.8616 | 0.8620 | 0.8620 |

## Stage 2: humanitarian category (official test set)

| Model | Seeds | Weighted F1 | Macro F1 | Weighted P | Weighted R | Accuracy |
|---|---|---|---|---|---|---|
| bertweet-base | 1 | 0.8632 | 0.7135 | 0.8717 | 0.8586 | 0.8586 |
| majority | 1 | 0.4386 | 0.0675 | 0.3488 | 0.5906 | 0.5906 |
| tfidf_lr | 1 | 0.8271 | 0.6467 | 0.8266 | 0.8286 | 0.8286 |

## End to end: full pipeline (tweets in both test sets)

| Model | Seeds | Weighted F1 | Macro F1 | Weighted P | Weighted R | Accuracy |
|---|---|---|---|---|---|---|
| bertweet-base -> bertweet-base | 1 | 0.8770 | 0.7316 | 0.8828 | 0.8741 | 0.8741 |
| tfidf_lr -> tfidf_lr | 1 | 0.8417 | 0.6752 | 0.8410 | 0.8446 | 0.8446 |
