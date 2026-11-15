#!/usr/bin/env bash
# Runs the whole experiment from start to finish.
#
#   bash scripts/run_all.sh /path/to/crisis_datasets_benchmarks
#
# Each step can also be run on its own; see README.md.
# On a single GPU, the BERTweet runs take most of the time. To get demo-ready
# models first, run with SEEDS="1", then later with SEEDS="2 3".

set -e

CRISISBENCH_DIR="$1"
SEEDS="${SEEDS:-1 2 3}"
MODEL="${MODEL:-vinai/bertweet-base}"
MODEL_SHORT="$(basename "$MODEL")"

if [ -z "$CRISISBENCH_DIR" ]; then
  echo "Usage: bash scripts/run_all.sh /path/to/crisis_datasets_benchmarks"
  exit 1
fi

echo "== 1. Preparing data =="
python src/prepare_data.py --crisisbench-dir "$CRISISBENCH_DIR"

echo "== 2. Baselines =="
python src/run_baselines.py
python src/evaluate_end_to_end.py --stage1-model tfidf_lr --stage1-seed 0 \
                                  --stage2-model tfidf_lr --stage2-seed 0

echo "== 3. $MODEL_SHORT, seeds: $SEEDS =="
for SEED in $SEEDS; do
  python src/train_transformer.py --stage 1 --seed "$SEED" --model-name "$MODEL"
  python src/train_transformer.py --stage 2 --seed "$SEED" --model-name "$MODEL"
  python src/evaluate_end_to_end.py --stage1-model "$MODEL_SHORT" --stage1-seed "$SEED" \
                                    --stage2-model "$MODEL_SHORT" --stage2-seed "$SEED"
done

echo "== 4. Summary tables =="
python src/summarize_results.py

echo "Done. Tables are in results/summary.md"
