# Triaging disaster tweets with a two-stage BERTweet pipeline

COSC 72/272 final project: Carys Kirlew, Joanna Shan, Ella Jacobs

This project categorizes disaster tweets so responders can find and prioritize actionable posts faster. Each tweet goes through two classifiers:

```
tweet ──> Stage 1: informative?
            ├─ no  ──> "Filtered out" tab (kept for review, never deleted)
            └─ yes ──> Stage 2: Which humanitarian category?
                         Requests or needs, infrastructure damage, missing or found people, ...
                         or "Other" (informative but non-humanitarian)
```

Both stages are fine-tuned BERTweet models trained only on CrisisBench (Alam et al., 2021). The demo is a simulated X-style feed of CrisisBench test tweets, with a Chrome extension that labels and filters them.

Everything runs from the terminal. All commands below are run from the project's root folder (the one containing `src/`, `server/`, and `scripts/`).

## Contents

```
src/
  config.py               shared settings: paths, label strings, colors
  crisisbench_io.py       reading/writing files, text cleanup
  prepare_data.py         official splits, leakage removal, end-to-end test set
  class_weights.py        weights for rare categories
  metrics.py              every metric and confusion-matrix plot
  run_outputs.py          saves each run's predictions and metrics the same way
  run_baselines.py        majority class and TF-IDF + logistic regression
  train_transformer.py    BERTweet (or RoBERTa) fine-tuning, one stage, one seed
  evaluate_end_to_end.py  scores the full pipeline
  summarize_results.py    mean ± std across seeds, report-ready tables
  significance_test.py    paired bootstrap test between two systems
  error_review.py         exports errors for manual cause tagging, then counts tags
  triage_pipeline.py      the two-stage pipeline used by the demo server
server/app.py             demo server: model API and the feed page
demo/feed/                the simulated "Y" feed (plain page, no model code)
extension/                the Chrome extension (labels, tabs, answer key)
scripts/run_all.sh        the whole experiment in one command
```

## 1. Setup

You need Python 3.10 or newer and a terminal (macOS, Linux, or WSL on Windows). Training needs a GPU to finish in a reasonable time. Everything else (data prep, baselines, the demo with TF-IDF models) runs on CPU.

**Get the code:**

```bash
git clone https://github.com/joannashan28/Crisis-Triage-Project.git
cd Crisis-Triage-Project
```

**Create a virtual environment and install the requirements:**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Run `source .venv/bin/activate` again in every new terminal before running any command below.

**Check that PyTorch can see the GPU** (skip this if you only run baselines):

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| mps:', torch.backends.mps.is_available())"
```

## 2. Get the data

Download CrisisBench v1.0 and extract it into a folder outside the repo, for example `~/data`:

```bash
mkdir -p ~/data && cd ~/data
wget https://crisisnlp.qcri.org/data/crisis_datasets_benchmarks/crisis_datasets_benchmarks_v1.0.tar.gz
tar -xzf crisis_datasets_benchmarks_v1.0.tar.gz
cd -                              # back to the project folder
```

On macOS, `wget` is not installed by default. Use `curl -O <same URL>` instead, or `brew install wget`.

The extracted folder is `~/data/crisis_datasets_benchmarks_v1.0` in this example. Use that path wherever the commands below say `/path/to/crisis_datasets_benchmarks_v1.0`.

Extracting CrisisBench into the repo's `data/` folder also works; in that case use `--crisisbench-dir data`.

CrisisBench is released under CC BY-NC-SA 4.0 (non-commercial use, with attribution). We don't redistribute it here; download it from the link above. `.gitignore` excludes it.

## 3. Run the experiment

You can run everything at once:

```bash
bash scripts/run_all.sh /path/to/crisis_datasets_benchmarks_v1.0
```

Or step by step, as below. Every script also explains itself with `--help`, for example `python src/train_transformer.py --help`.

### 3a. Prepare the data

```bash
python src/prepare_data.py --crisisbench-dir /path/to/crisis_datasets_benchmarks_v1.0
```

The script writes `data/processed/` and `data/processed/data_report.json`, which records every count.

**Check this on the first run:** the script verifies that the label spellings in `config.py` match the real files. If one doesn't match, it stops and prints the labels it actually found. It also works out each tweet's disaster type from its event name (for example, `2015_nepal_earthquake` becomes earthquake), using keyword rules in `config.py`, and prints any event names that matched no rule. If one of those clearly has a type, add a keyword to `DISASTER_TYPE_RULES` and rerun. The full event-to-type mapping is saved in `data_report.json`.

### 3b. Baselines (minutes, no GPU)

```bash
python src/run_baselines.py
```

### 3c. BERTweet (GPU)

Do a quick check first, so a bug doesn't cost you a full run:

```bash
python src/train_transformer.py --stage 1 --seed 1 --epochs 1 --max-train-examples 2000
```

Then run the real thing, once per seed:

```bash
for SEED in 1 2 3; do
  python src/train_transformer.py --stage 1 --seed $SEED
  python src/train_transformer.py --stage 2 --seed $SEED
done
```

Stage 1 takes longer than Stage 2 because its training set is roughly twice as large.

**Long runs.** Training can take hours, and closing the terminal or losing an SSH connection kills a normal command. Run it inside `tmux` (or `screen`) so it keeps going:

```bash
tmux new -s train                 # start a session
# run the training commands inside it, then press Ctrl+b, then d, to detach
tmux attach -t train              # come back later
```

Or run it in the background and write the output to a log file:

```bash
nohup bash -c 'for SEED in 1 2 3; do
  python src/train_transformer.py --stage 1 --seed $SEED
  python src/train_transformer.py --stage 2 --seed $SEED
done' > train.log 2>&1 &
tail -f train.log                 # watch progress (Ctrl+c stops watching, not training)
```

### 3d. Score the full pipeline

```bash
python src/evaluate_end_to_end.py --stage1-model bertweet-base --stage1-seed 1 \
                                  --stage2-model bertweet-base --stage2-seed 1
python src/evaluate_end_to_end.py --stage1-model tfidf_lr --stage1-seed 0 \
                                  --stage2-model tfidf_lr --stage2-seed 0
```

Model names are the run folder names under `runs/`. Baselines are deterministic, so they only have seed 0. To try a lower Stage 1 cutoff, add `--threshold 0.3`. The threshold sweep in `metrics.json` shows the trade-off across all cutoffs.

### 3e. Tables, significance, and error analysis

```bash
python src/summarize_results.py           # results/summary.md and CSVs

python src/significance_test.py \
    --a runs/bertweet-base/stage2/seed1/predictions.tsv \
    --b runs/tfidf_lr/stage2/seed0/predictions.tsv \
    --metric macro_f1

python src/error_review.py export \
    --run runs/end_to_end/stage1-bertweet-base-seed1__stage2-bertweet-base-seed1
# fill in the cause_tag column of results/error_review.csv, then:
python src/error_review.py summarize --file results/error_review.csv
```

## 4. What each run produces

```
runs/<model>/stage<1|2>/seed<N>/
    predictions.tsv        every test tweet: gold, predicted, confidence
    metrics.json           overall, per-class, by disaster type, by source, by length
    confusion_matrix.png
    model/                 the saved model (used by the demo)
runs/end_to_end/<stage1 run>__<stage2 run>/
    predictions.tsv        includes an error_type column for each tweet
    metrics.json           also has per-category recall, Stage 1 losses, threshold sweep
    confusion_matrix.png
```

## Trained models

Trained models are not in this repo; each BERTweet model is over 500 MB, which is larger than GitHub's file-size limit. The trained models can be rebuilt from the steps in section 3. The demo needs the seed 1 models for both stages, the end-to-end run for seed 1, and `data/processed/` (from section 3a).

## 5. Run the demo

NOTE: Before we presented, we decided that we wanted to substantially change the Chrome extension for our demo. While the original code to run the extension is included in the `main` branch and still works, we have a separate branch called `day-of-demo` that includes the code for the new extension that we ended up showing in class.

Class Presentation Link: https://docs.google.com/presentation/d/1mo2jdzuSjY1Q4LVgcBZEr2QoAeO6dOwUu-Edknt6ZP8/edit?usp=sharing

**Start the server** (from the project root, with the virtual environment active):

```bash
python server/app.py \
    --stage1 runs/bertweet-base/stage1/seed1/model \
    --stage2 runs/bertweet-base/stage2/seed1/model \
    --errors runs/end_to_end/stage1-bertweet-base-seed1__stage2-bertweet-base-seed1/predictions.tsv
```

Leave this terminal open; the server runs until you press `Ctrl+c`.

If BERTweet isn't trained yet, the TF-IDF models work the same way. Point `--stage1` and `--stage2` at `runs/tfidf_lr/stage1/seed0/model` and `runs/tfidf_lr/stage2/seed0/model`, and set `--errors` to the TF-IDF end-to-end run. Use the `--errors` file that matches the loaded models; otherwise the "Known mistakes" tab will show the wrong tweets.

**Load the extension (once):** The extension runs in Chrome on the computer where you open the demo. Open `chrome://extensions`, turn on Developer mode, click **Load unpacked**, and choose the `extension/` folder.

**Open** http://127.0.0.1:8000 in Chrome.

The interactive parts:

- **Disaster picker** narrows the feed to one event, such as one hurricane or one earthquake.
- **Category tabs** filter by Stage 2 label, with counts.
- **Filtered out** shows what Stage 1 rejected, so missed requests for help can still be found.
- **Show answer key** puts the annotators' labels next to the model's and highlights disagreements.
- **Known mistakes** shows only test tweets the pipeline got wrong, which is useful for the failure-case part of the demo.

## 6. Design decisions

**Data and leakage.** We use CrisisBench's official English splits. The two tasks were split separately, so the same tweet can be in one task's test set and the other task's training set. We remove from all training sets any tweet whose ID or normalized text appears in any dev or test set, and from dev sets anything in a test set. Test sets are unchanged, so our test results are on the same tweets as published CrisisBench numbers. Removing leakage cut the training sets by about 17% (Stage 1) and 30% (Stage 2), though, so our models train on less data than the published CrisisBench baselines. Exact counts are in `data/processed/data_report.json`.

**Disaster types.** CrisisBench does not provide a disaster-type label, so we assign one from each event name using keyword rules in `config.py` (for example, `2015_nepal_earthquake` becomes earthquake). Typhoons and cyclones are grouped with hurricanes. The mapping for all 61 events is saved in `data_report.json`.

**End-to-end gold label.** If annotators marked a tweet "not informative," the correct pipeline output is "not actionable." Otherwise it is the humanitarian label, with "not humanitarian" also counted as "not actionable." `data_report.json` counts how often the two sets of annotations conflict.

**Stage 2 training data.** Stage 2 trains on the full humanitarian training set, including "not humanitarian." That way, if Stage 1 lets noise through, Stage 2 can still send it to "Other" rather than force it into a category.

**Class weights.** Stage 2 uses square-root-balanced class weights (see `class_weights.py`), and Stage 1 uses none. This can be changed with `--class-weighting`.

**Model selection.** For each run we keep the epoch with the best dev weighted F1. The test set is scored once.

## 7. Limitations to keep in mind

- Disaster types appear in both training and test, so these results do not show how the model handles a new kind of disaster.
- The end-to-end set is only the 3,430 test tweets that appear in both test sets, so rare categories have few end-to-end examples.
- Disaster types come from our keyword rules on event names, not from CrisisBench annotations.
- CrisisBench is English only, and some disaster types have very few tweets.

## Generative AI disclosure

We used Claude Opus 5.5 for a wide variety of tasks related to this project. Specifically, Claude helped draft the original file tree so we could figure out what modules we needed for the tweet classifier to come to fruition. After we figured out the entire pipeline and what models/tests we wanted to run, we prompted Claude to helped us write the original/draft code files. In regards to the data, we decided which tweets we wanted to remove from the training/test sets and prompted Claude to help us write the code that would allow us to clean up the data. We examined and edited all the code Claude generated for accuracy in order to ensure it was doing what we intended for it to do. We decided exactly how we wanted the Chrome extension to look and run and then prompted Claude to help us write the code for the extension (have two versions of the extension, an original and an improved version for the demo) since none of us have ever done something like that before. 

## References

- Alam, F., Sajjad, H., Imran, M., & Ofli, F. (2021). CrisisBench: Benchmarking crisis-related social media datasets for humanitarian information processing. *ICWSM*. Data and code: https://github.com/firojalam/crisis_datasets_benchmarks
- Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. *EMNLP: System Demonstrations*. Model: https://huggingface.co/vinai/bertweet-base
