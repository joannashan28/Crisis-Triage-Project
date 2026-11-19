# Triaging Disaster Tweets with a Two-Stage BERTweet Pipeline

COSC 72/272 final project · Carys Kirlew, Joanna Shan, Ella Jacobs

During a disaster, the posts responders can act on (requests for help, damage reports, missing people) are buried under sympathy, jokes and unrelated chatter. This project sorts disaster tweets into responder-facing categories using two classifiers:

```
tweet ──> Stage 1: informative?
            ├─ no  ──> "Filtered out" (kept for review, never deleted)
            └─ yes ──> Stage 2: which humanitarian category?
                         requests or needs, infrastructure damage, missing or found people, ...
                         or "Other" (informative, but no humanitarian category)
```

Both stages are fine-tuned [BERTweet](https://huggingface.co/vinai/bertweet-base) models trained on [CrisisBench](https://github.com/firojalam/crisis_datasets_benchmarks) (Alam et al., 2021). We compare them against a majority-class baseline and TF-IDF + logistic regression, score each stage on its own, and score the full pipeline end to end.

- **Slides:** https://docs.google.com/presentation/d/1mo2jdzuSjY1Q4LVgcBZEr2QoAeO6dOwUu-Edknt6ZP8/edit?usp=sharing
- **Demo video:** [TODO: add link]
- **Branches:** `main` has the experiments and the original Chrome-extension demo. `day-of-demo` has the standalone triage page we showed in class (see [Run the demo](#5-run-the-demo)).

## Results

Test-set results for one run (BERTweet seed 1; the baselines are deterministic). Full tables are in [`results/summary.md`](results/summary.md), and per-category F1 is in [`results/per_class_f1.csv`](results/per_class_f1.csv).

| Task | Model | Weighted F1 | Macro F1 |
|---|---|---|---|
| Stage 1: informative? | Majority | 0.449 | 0.375 |
| | TF-IDF + LR | 0.861 | 0.855 |
| | **BERTweet** | **0.886** | **0.881** |
| Stage 2: category (11 classes) | Majority | 0.439 | 0.068 |
| | TF-IDF + LR | 0.827 | 0.647 |
| | **BERTweet** | **0.863** | **0.714** |
| End to end (3,430 tweets in both test sets) | TF-IDF → TF-IDF | 0.842 | 0.675 |
| | **BERTweet → BERTweet** | **0.877** | **0.732** |

Takeaways:
- **BERTweet beats TF-IDF on every summary metric.** The gain is largest on macro F1, so most of it comes from the smaller categories. Examples: *response efforts* (Stage 2 F1 0.27 → 0.43) and *displaced or evacuations* (0.40 → 0.53).
- ***Requests or needs*, the most important category for triage, reaches 0.93 F1.**
- **The weakest category is *missing or found people*** (about 0.46–0.54 F1), where BERTweet does no better than TF-IDF. It is rare in the training data and overlaps with requests and affected individuals.
- **These results come from a single seed.** The scripts support seeds 2 and 3 and a paired bootstrap test (see [step 3e](#3e-tables-significance-and-error-analysis)).

## Contents

```
src/
  config.py               shared settings: paths, label strings, disaster-type rules, colors
  crisisbench_io.py       reading/writing TSV files, text normalization
  prepare_data.py         official splits, leakage removal, end-to-end test set
  class_weights.py        none / balanced / sqrt-balanced class weights
  metrics.py              all metrics and confusion-matrix plots
  run_outputs.py          saves every run's predictions and metrics in the same format
  run_baselines.py        majority class and TF-IDF + logistic regression
  train_transformer.py    BERTweet (or RoBERTa) fine-tuning, one stage, one seed
  evaluate_end_to_end.py  scores the full pipeline, error types, threshold sweep
  summarize_results.py    mean ± std across seeds -> results/
  significance_test.py    paired bootstrap test between two systems
  error_review.py         exports errors for manual cause tagging, then counts the tags
  triage_pipeline.py      the two-stage pipeline used by the demo server
server/app.py             demo server: classification API + pages
demo/feed/                simulated X-style feed ("Y") of test tweets
extension/                Chrome extension that labels and filters the feed
scripts/run_all.sh        the whole experiment in one command
results/                  summary tables used in the report
```

## 1. Setup

You need Python 3.10+ and a terminal (macOS, Linux, or WSL). Training BERTweet needs a GPU (NVIDIA CUDA or Apple Silicon MPS). Data prep, baselines, and the demo with TF-IDF models run on a CPU.

```bash
git clone https://github.com/joannashan28/Crisis-Triage-Project.git
cd Crisis-Triage-Project
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Activate the virtual environment again in every new terminal. All commands below are run from the project root.

To check that PyTorch can see your GPU:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| mps:', torch.backends.mps.is_available())"
```

## 2. Get the data

Download CrisisBench v1.0 and extract it somewhere outside the repo, for example `~/data`:

```bash
mkdir -p ~/data && cd ~/data
curl -O https://crisisnlp.qcri.org/data/crisis_datasets_benchmarks/crisis_datasets_benchmarks_v1.0.tar.gz
tar -xzf crisis_datasets_benchmarks_v1.0.tar.gz
cd -
```

The commands below refer to the extracted folder as `/path/to/crisis_datasets_benchmarks_v1.0`. You can also extract it into the repo's `data/` folder and pass `--crisisbench-dir data`. `.gitignore` excludes it either way.

**License:** CrisisBench is released under CC BY-NC-SA 4.0 (non-commercial use, with attribution). We do not redistribute it.

## 3. Run the experiment

To run everything in one command (data prep → baselines → BERTweet for seeds 1, 2, 3 → end-to-end scoring → summary tables):

```bash
bash scripts/run_all.sh /path/to/crisis_datasets_benchmarks_v1.0
```

To run only seed 1, for example to get demo-ready models quickly:

```bash
SEEDS="1" bash scripts/run_all.sh /path/to/crisis_datasets_benchmarks_v1.0
```

The steps below do the same thing one at a time. Every script explains its options with `--help`.

### 3a. Prepare the data

```bash
python src/prepare_data.py --crisisbench-dir /path/to/crisis_datasets_benchmarks_v1.0
```

This writes `data/processed/` and `data/processed/data_report.json`, which records every count (sizes, leakage removed, label counts, annotation conflicts, and the event-to-disaster-type mapping). The script stops if the label strings in `config.py` don't match the data. It also prints any event names that match no disaster-type rule; to fix one, add a keyword to `DISASTER_TYPE_RULES` in `config.py`.

### 3b. Baselines (minutes, CPU)

```bash
python src/run_baselines.py
```

### 3c. BERTweet (GPU)

Do a quick check first, so a bug doesn't cost you a full run:

```bash
python src/train_transformer.py --stage 1 --seed 1 --epochs 1 --max-train-examples 2000
```

Then train both stages for each seed:

```bash
for SEED in 1 2 3; do
  python src/train_transformer.py --stage 1 --seed $SEED
  python src/train_transformer.py --stage 2 --seed $SEED
done
```

Stage 1 takes longer because its training set is about twice as large. Full runs can take hours, so run them inside `tmux`, or in the background with `nohup ... > train.log 2>&1 &`, so that closing the terminal doesn't kill training.

To compare against a general-domain model with the same architecture, add `--model-name roberta-base`.

### 3d. Score the full pipeline

```bash
python src/evaluate_end_to_end.py --stage1-model bertweet-base --stage1-seed 1 \
                                  --stage2-model bertweet-base --stage2-seed 1
python src/evaluate_end_to_end.py --stage1-model tfidf_lr --stage1-seed 0 \
                                  --stage2-model tfidf_lr --stage2-seed 0
```

Model names are the folder names under `runs/`, and the baselines only have seed 0. You can mix models across stages, for example a BERTweet Stage 1 with a TF-IDF Stage 2. The default Stage 1 cutoff is τ = 0.5; `--threshold 0.3` scores a different cutoff, and `metrics.json` always includes a sweep over τ = 0.1 … 0.9.

### 3e. Tables, significance, and error analysis

```bash
python src/summarize_results.py          # writes results/summary.md, summary.csv, per_class_f1.csv
```

```bash
python src/significance_test.py \
    --a runs/bertweet-base/stage2/seed1/predictions.tsv \
    --b runs/tfidf_lr/stage2/seed0/predictions.tsv \
    --metric macro_f1
```

```bash
python src/error_review.py export \
    --run runs/end_to_end/stage1-bertweet-base-seed1__stage2-bertweet-base-seed1
```

After exporting, fill in the `cause_tag` column of `results/error_review.csv`, then count the tags:

```bash
python src/error_review.py summarize --file results/error_review.csv
```

## 4. What each run produces

```
runs/<model>/stage<1|2>/seed<N>/
    predictions.tsv        every test tweet: gold, predicted, confidence
    metrics.json           overall, per-class, by disaster type / source / length, training settings
    confusion_matrix.png   row-normalized, with raw counts
    model/                 the saved model (used by the demo)
runs/end_to_end/<stage1 run>__<stage2 run>/
    predictions.tsv        adds an error_type column for each tweet
    metrics.json           adds per-category recall, triage rates, threshold sweep
    confusion_matrix.png
```

Trained models are not in the repo (each BERTweet model is over 500 MB). Rebuild them with section 3. The demo needs the seed-1 models for both stages, the matching end-to-end run, and `data/processed/`.

## 5. Run the demo

Start the server:

```bash
python server/app.py \
    --stage1 runs/bertweet-base/stage1/seed1/model \
    --stage2 runs/bertweet-base/stage2/seed1/model \
    --errors runs/end_to_end/stage1-bertweet-base-seed1__stage2-bertweet-base-seed1/predictions.tsv
```

Leave the server running in that terminal (`Ctrl+C` stops it). If BERTweet isn't trained yet, point `--stage1` and `--stage2` at `runs/tfidf_lr/stage{1,2}/seed0/model` and set `--errors` to the TF-IDF end-to-end run. The `--errors` file must match the models you load, or the "Known mistakes" view will show the wrong tweets.

There are two versions of the demo.

**Class demo (`day-of-demo` branch, no extension needed).** Run `git checkout day-of-demo`, start the server, and open http://127.0.0.1:8000. The page lets you:
- type or paste any post and classify it, or click **Try an example** to cycle through 10 posts we wrote for the demo (not from CrisisBench). They include clear cases and known failure cases.
- see the test tweets in a **Triage** view grouped by category, plus a **Filtered out** tab sorted by how close each tweet came to the threshold
- filter by disaster, or show only **Known mistakes**
- move the **τ slider** to see the filtering trade-off live. Reported numbers always use τ = 0.5; the slider is for exploring.
- open a **Summary** tab with the loaded model's end-to-end test scores

The plain feed for the extension is at http://127.0.0.1:8000/feed/ on this branch.

**Original demo (`main` branch, Chrome extension).** Open `chrome://extensions`, turn on Developer mode, click **Load unpacked**, and choose the `extension/` folder. Then open http://127.0.0.1:8000 in Chrome. The extension labels each tweet with its category and confidence, and adds:
- category tabs with counts
- a disaster picker
- a **Filtered out** tab
- a **Show answer key** switch, which puts the annotators' labels next to the model's and highlights disagreements
- a **Known mistakes** view

## 6. Design decisions

- **Leakage removal.** CrisisBench split its two tasks independently, so a tweet can be in one task's test set and the other task's training set. We remove from both training sets any tweet whose ID or normalized text (lowercased, with links, mentions, "RT" and punctuation removed) appears in any dev or test set. We also remove from the dev sets anything in a test set. Test sets are unchanged, so our test results are on the same tweets as published CrisisBench numbers. This cut the training sets by about 17% (Stage 1) and 30% (Stage 2).
- **End-to-end gold label.** If the annotators said "not informative", the correct output is *not actionable*. Otherwise it is the humanitarian label, with "not humanitarian" also counted as *not actionable*.
- **Stage 2 trains on "not humanitarian" too.** If Stage 1 lets noise through, Stage 2 can still send it to "Other" instead of forcing it into a category.
- **Class weights.** Stage 2 uses square-root-balanced weights, so a class with 100× fewer examples gets 10× the weight. This boosts rare categories without flooding them with false positives. Stage 1 is only mildly imbalanced and uses no weights. The TF-IDF baseline uses the same weights, so the comparison is fair. Change this with `--class-weighting`.
- **Model selection.** BERTweet: lr 2e-5, batch 32, 3 epochs, 10% warmup, max 128 tokens, BERTweet's tweet normalization on. We keep the epoch with the best dev weighted F1. TF-IDF: C chosen from {0.25, 1, 4} on dev. The test set is scored once.
- **Metrics.** Weighted F1 is the primary metric (comparable to CrisisBench). Macro F1 is the secondary metric because every category counts equally, and the rare categories matter most to responders.
- **Disaster types.** CrisisBench has no disaster-type label, so we assign one from each of the 61 event names using keyword rules in `config.py`. Typhoons and cyclones are grouped with hurricanes. We use these types only for result breakdowns.

## 7. Limitations

- One BERTweet seed so far, so there is no variance estimate yet.
- Disaster types appear in both training and test, so the results don't show how the model handles a new kind of disaster.
- The end-to-end set is only the 3,430 tweets in both test sets, so rare categories have few end-to-end examples.
- Disaster types come from our keyword rules, not from annotations.
- English only. Some disaster types have very few tweets.
- Single-label scoring penalizes tweets that reasonably belong to two categories, such as a missing-person post that is also a request.

## Generative AI disclosure

We used Claude Opus 5.5 for a wide variety of tasks related to this project. Specifically, Claude helped draft the original file tree so we could figure out what modules we needed for the tweet classifier to come to fruition. After we figured out the entire pipeline and what models/tests we wanted to run, we prompted Claude to helped us write the original/draft code files. In regards to the data, we decided which tweets we wanted to remove from the training/test sets and prompted Claude to help us write the code that would allow us to clean up the data. We examined and edited all the code Claude generated for accuracy in order to ensure it was doing what we intended for it to do. We decided exactly how we wanted the Chrome extension to look and run and then prompted Claude to help us write the code for the extension (have two versions of the extension, an original and an improved version for the demo) since none of us have ever done something like that before.

## References

- Alam, F., Sajjad, H., Imran, M., & Ofli, F. (2021). CrisisBench: Benchmarking crisis-related social media datasets for humanitarian information processing. *ICWSM*. https://github.com/firojalam/crisis_datasets_benchmarks
- Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. *EMNLP: System Demonstrations*. https://huggingface.co/vinai/bertweet-base
