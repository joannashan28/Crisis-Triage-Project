# Triaging disaster tweets with a two-stage BERTweet pipeline

COSC 72/272 final project: Ella Jacobs, Carys Kirlew, Joanna Shan

This project sorts disaster tweets so responders can find actionable posts faster. Each tweet goes through two classifiers:

```
tweet ──> Stage 1: informative?
            ├─ no  ──> "Filtered out" tab (kept for review, never deleted)
            └─ yes ──> Stage 2: which humanitarian category?
                         requests or needs, infrastructure damage, missing or found people, ...
                         or "Other" (informative, but no humanitarian category)
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

You need Python 3.10 or newer and a terminal (macOS, Linux, or WSL on Windows). Training needs a GPU to finish in a reasonable time. `train_transformer.py` picks the device automatically, in this order:

| Device | Hardware | Notes |
|---|---|---|
| `cuda` | NVIDIA GPU (Linux, Windows, lab or cluster machines) | fastest; uses mixed precision |
| `mps` | Apple Silicon Mac (M1/M2/M3/M4) | works, full precision, slower than a good NVIDIA GPU |
| `cpu` | anything else, including Intel Macs | works but is very slow for BERTweet; the script prints a warning |

Everything else (data prep, baselines, the demo with TF-IDF models) runs on CPU.

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

On a Mac, `mps: True` is what you want. On an NVIDIA machine, you want `cuda: True` (`nvidia-smi` should also list the GPU). If both print `False`, training will fall back to CPU. On an NVIDIA machine, reinstall PyTorch with the CUDA build that matches your driver (see https://pytorch.org/get-started/locally/) and check again. When training starts, the script prints `Using device: ...`, so you can confirm which one it picked.

**Training on a remote machine or school cluster?** Connect with `ssh`, repeat the steps above there, and see the "Long runs" note in section 3c. To use the demo from your own laptop, see the port-forwarding note in section 5.

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

The script writes `data/processed/` and `data/processed/data_report.json`, which records every count. Check it once and keep it for the report.

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

Stage 1 takes longer than Stage 2 because its training set is roughly twice as large. The script prints the elapsed time every 50 steps, so you can estimate the total from the first epoch. If time is short, finish seed 1 for both stages first; that's enough for the demo.

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

On an NVIDIA machine, use `nvidia-smi` in another terminal to confirm the GPU is busy. **On a Mac**, a sleeping laptop pauses training; put `caffeinate -i` in front of the command (for example `caffeinate -i python src/train_transformer.py --stage 1 --seed 1`) and keep the laptop plugged in. On a shared cluster that uses a job scheduler such as Slurm, put the commands in a batch script and submit it with `sbatch` instead.

The comparison with general-text pretraining is optional (same architecture as BERTweet):

```bash
python src/train_transformer.py --stage 2 --seed 1 --model-name roberta-base
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

If you trained on a remote machine, copy `runs/` and `results/` to your own computer before the session or account expires. For example, from your laptop:

```bash
scp -r user@remote-host:~/Crisis-Triage-Project/runs ./
scp -r user@remote-host:~/Crisis-Triage-Project/results ./
```

(`rsync -av` works too and can resume an interrupted copy.)

## Trained models

Trained models are not in this repo; each BERTweet model is over 500 MB, past GitHub's file-size limit. Rebuild them with the steps in section 3. The demo needs the seed 1 models for both stages, the end-to-end run for seed 1, and `data/processed/` (from section 3a).

## 5. Run the demo

**Start the server** (from the project root, with the virtual environment active):

```bash
python server/app.py \
    --stage1 runs/bertweet-base/stage1/seed1/model \
    --stage2 runs/bertweet-base/stage2/seed1/model \
    --errors runs/end_to_end/stage1-bertweet-base-seed1__stage2-bertweet-base-seed1/predictions.tsv
```

Leave this terminal open; the server runs until you press `Ctrl+c`.

If BERTweet isn't trained yet, the TF-IDF models work the same way. Point `--stage1` and `--stage2` at `runs/tfidf_lr/stage1/seed0/model` and `runs/tfidf_lr/stage2/seed0/model`, and set `--errors` to the TF-IDF end-to-end run. Use the `--errors` file that matches the loaded models; otherwise the "Known mistakes" tab will show the wrong tweets.

**Load the extension (once):** the extension runs in Chrome on the computer where you open the demo. Open `chrome://extensions`, turn on Developer mode, click **Load unpacked**, and choose the `extension/` folder.

**Open** http://127.0.0.1:8000 in Chrome.

**Server on a remote machine?** If the server runs over SSH, forward its port to your laptop from a second terminal on the laptop, then open the same address in your local Chrome:

```bash
ssh -L 8000:127.0.0.1:8000 user@remote-host
```

The `extension/` folder and the repo copy you load it from must be on your laptop too (clone the repo there, or `scp` the folder).

The interactive parts:

- **Disaster picker** narrows the feed to one event, such as one hurricane or one earthquake.
- **Category tabs** filter by Stage 2 label, with counts.
- **Filtered out** shows what Stage 1 rejected, so missed requests for help can still be found.
- **Show answer key** puts the annotators' labels next to the model's and highlights disagreements.
- **Known mistakes** shows only test tweets the pipeline got wrong, which is useful for the failure-case part of the demo.

Record a backup video of the demo in case the live version breaks during the presentation.

## 6. Design decisions

These are documented here so the report, code, and demo stay consistent.

**Data and leakage.** We use CrisisBench's official English splits. The two tasks were split separately, so the same tweet can be in one task's test set and the other task's training set. We remove from all training sets any tweet whose ID or normalized text appears in any dev or test set, and from dev sets anything in a test set. Test sets are unchanged, so our test results are on the same tweets as published CrisisBench numbers. Removing leakage cut the training sets by about 17% (Stage 1) and 30% (Stage 2), though, so our models train on less data than the published CrisisBench baselines. Exact counts are in `data/processed/data_report.json`.

**Disaster types.** CrisisBench does not provide a disaster-type label, so we assign one from each event name using keyword rules in `config.py` (for example, `2015_nepal_earthquake` becomes earthquake). Typhoons and cyclones are grouped with hurricanes. The mapping for all 61 events is saved in `data_report.json`.

**End-to-end gold label.** If annotators marked a tweet "not informative," the correct pipeline output is "not actionable." Otherwise it is the humanitarian label, with "not humanitarian" also counted as "not actionable." `data_report.json` counts how often the two sets of annotations conflict.

**Stage 2 training data.** Stage 2 trains on the full humanitarian training set, including "not humanitarian." That way, if Stage 1 lets noise through, Stage 2 can still send it to "Other" rather than force it into a category.

**Class weights.** Stage 2 uses square-root-balanced class weights (see `class_weights.py`), and Stage 1 uses none. This can be changed with `--class-weighting`.

**Model selection.** For each run we keep the epoch with the best dev weighted F1. The test set is scored once.

**Settings.** BERTweet uses learning rate 2e-5, batch size 32, 3 epochs, 10% warmup, weight decay 0.01, max length 128, and mixed precision on NVIDIA GPUs only (Apple Silicon and CPU runs use full precision). Every run's actual settings, including batch size, are saved in its `metrics.json`; if a run ran out of GPU memory and was redone with `--batch-size 16`, its `metrics.json` records that.

**Changes from the proposal:**

- The single-stage comparison was dropped; we built only the two-stage pipeline.
- BERTweet is the only main model. The baselines are majority class and TF-IDF with logistic regression, with RoBERTa as an optional comparison.
- The held-out disaster-type split was dropped. We report results by disaster type on the official test set instead.
- No hand-labeled recent-disaster set and no scraper. Training, testing, and the demo use only CrisisBench.

## 7. Limitations to keep in mind

- Disaster types appear in both training and test, so these results do not show how the model handles a new kind of disaster.
- The end-to-end set is only the 3,430 test tweets that appear in both test sets, so rare categories have few end-to-end examples.
- Disaster types come from our keyword rules on event names, not from CrisisBench annotations.
- CrisisBench is English only, and some disaster types have very few tweets.

## Generative AI disclosure

We used Claude (Anthropic, Claude Opus 5.5, via claude.ai) substantially in this project.

**What it produced**
- Nearly all of the code in `src/`, `server/`, `demo/`, `extension/`, and `scripts/`: data preparation and leakage removal, the baselines, BERTweet fine-tuning, evaluation, the summary, significance, and error-review scripts, the demo server, the simulated feed, and the Chrome extension.
- The first draft of this README.
- A draft outline of our presentation slides.
- Fixes for errors we hit while running the code: the disaster-type mapping, Apple Silicon GPU support, a GPU out-of-memory crash, and the feed's "Show more" behavior.

**Decisions the team made**
- [FILL IN: e.g., the two-stage design, dropping the single-stage comparison, using only BERTweet plus two baselines, using only CrisisBench data, skipping the held-out disaster-type split]

**How we checked it**
- [FILL IN: what you ran and checked, e.g., data counts in data_report.json against our proposal's exploration, the leakage-removal counts, the disaster-type mapping for all 61 events]
- [FILL IN: who reviewed which files, and anything you changed or rejected]
- [FILL IN: how you checked results, e.g., BERTweet against the baselines]
- The interpretation of results and the analysis in our report were written by the team. [Edit if not true.]

**Earlier use (proposal stage)**
- Claude Code was used to build an early prototype with fabricated posts from the LA fire. That prototype is not part of the final project.
- Claude helped plan our division of labor, and wrote pseudocode for our initial data exploration.

## References

- Alam, F., Sajjad, H., Imran, M., & Ofli, F. (2021). CrisisBench: Benchmarking crisis-related social media datasets for humanitarian information processing. *ICWSM*. Data and code: https://github.com/firojalam/crisis_datasets_benchmarks
- Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. *EMNLP: System Demonstrations*. Model: https://huggingface.co/vinai/bertweet-base
- Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP: System Demonstrations*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12.
- The original datasets that CrisisBench combines are listed, with citations, in the CrisisBench README. Cite them too.
