"""
Fine-tune a transformer (BERTweet by default) for one stage and one seed.

How training works:
  - Tweets are tokenized once, up front. For BERTweet we turn on its
    built-in tweet normalization, which replaces links with HTTPURL,
    @mentions with @USER, and emoji with text (the same preprocessing
    BERTweet was pretrained with).
  - Each epoch we train on the training set, then score the dev set. We keep
    the epoch with the best dev weighted F1 (our primary metric) and save
    that model. The test set is only touched once, at the very end.
  - Stage 2 uses class-weighted cross-entropy loss (see class_weights.py) so
    the rare categories are not ignored.

Usage (run once per seed, e.g. seeds 1, 2, 3):
    python src/train_transformer.py --stage 1 --seed 1
    python src/train_transformer.py --stage 2 --seed 1

    # Optional comparison: same architecture, general-text pretraining
    python src/train_transformer.py --stage 2 --seed 1 --model-name roberta-base

    # Quick check that everything runs before a full training run
    python src/train_transformer.py --stage 1 --seed 1 --epochs 1 --max-train-examples 2000
"""

import argparse
import contextlib
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

import config
from class_weights import WEIGHTING_CHOICES, compute_class_weights
from crisisbench_io import load_processed_split, save_json
from metrics import compute_metrics
from run_outputs import run_directory, save_run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-name", default=config.DEFAULT_TRANSFORMER)
    parser.add_argument("--data-dir", default=config.PROCESSED_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128,
                        help="BERTweet supports at most 128 tokens.")
    parser.add_argument("--class-weighting", default=None, choices=WEIGHTING_CHOICES,
                        help="Default: none for Stage 1, sqrt_balanced for Stage 2.")
    parser.add_argument("--max-train-examples", type=int, default=None,
                        help="Only for quick debugging. Do not use for reported results.")
    return parser.parse_args()


def set_seed(seed):
    """Make runs repeatable: same seed, same shuffling and initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    """Pick the fastest hardware available:
      cuda  NVIDIA GPU (Colab, lab machines), with mixed precision
      mps   Apple Silicon GPU (M1/M2/M3/M4 Macs), full precision
      cpu   anything else (very slow for BERTweet)
    Returns the device and whether to use mixed precision."""
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    if torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), False


def precision_context(device, use_mixed_precision):
    """Mixed precision (float16) is only used on NVIDIA GPUs. Everywhere
    else this returns a context that does nothing."""
    if use_mixed_precision:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def short_model_name(model_name):
    """'vinai/bertweet-base' -> 'bertweet-base' (used for folder names)."""
    parts = model_name.split("/")
    return parts[-1]


def tokenizer_settings(model_name):
    """BERTweet needs its normalization switched on, and that option only
    exists in its slow (Python) tokenizer."""
    lowered = model_name.lower()
    if "bertweet" in lowered:
        return {"normalization": True, "use_fast": False}
    return {}


def load_tokenizer(model_name):
    settings = tokenizer_settings(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **settings)
    return tokenizer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TweetDataset(Dataset):
    """Holds already-tokenized tweets and their label numbers."""

    def __init__(self, encodings, label_ids):
        self.encodings = encodings
        self.label_ids = label_ids

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, index):
        item = {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.label_ids[index],
        }
        return item


def tokenize_texts(tokenizer, texts, max_length):
    """Tokenize in chunks so we can print progress (BERTweet's tokenizer is
    slow, and the Stage 1 training set has over 100,000 tweets)."""
    encodings = {"input_ids": [], "attention_mask": []}
    chunk_size = 2000

    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        chunk_encodings = tokenizer(chunk, truncation=True, max_length=max_length)
        encodings["input_ids"].extend(chunk_encodings["input_ids"])
        encodings["attention_mask"].extend(chunk_encodings["attention_mask"])

        done = min(start + chunk_size, len(texts))
        if done % 20000 == 0 or done == len(texts):
            print("    tokenized " + str(done) + " / " + str(len(texts)))

    return encodings


def build_dataset(rows, tokenizer, label_to_id, max_length):
    texts = []
    label_ids = []
    for row in rows:
        texts.append(row["text"])
        label_ids.append(label_to_id[row["label"]])
    encodings = tokenize_texts(tokenizer, texts, max_length)
    return TweetDataset(encodings, label_ids)


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------

def free_gpu_memory(device):
    """Give unused GPU memory back. On Apple Silicon the GPU shares memory
    with the rest of the Mac, and memory left over from scoring the dev set
    can otherwise pile up until training runs out of memory."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def predict_probabilities(model, data_loader, device, use_mixed_precision):
    """Run the model over a data loader and return one row of class
    probabilities per tweet (as a numpy array)."""
    model.eval()
    all_probabilities = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with precision_context(device, use_mixed_precision):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            logits = logits.float()
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = probabilities.cpu()
            all_probabilities.append(probabilities.numpy())

    free_gpu_memory(device)
    return np.concatenate(all_probabilities, axis=0)


def probabilities_to_predictions(probabilities, labels_in_order):
    """Turn probability rows into predicted labels and confidences."""
    predicted = []
    confidences = []
    for probability_row in probabilities:
        best_index = int(np.argmax(probability_row))
        predicted.append(labels_in_order[best_index])
        confidences.append(float(probability_row[best_index]))
    return predicted, confidences


def train_one_epoch(model, data_loader, optimizer, scheduler, scaler, loss_function,
                    device, use_mixed_precision, epoch_number):
    model.train()
    total_loss = 0.0
    start = time.time()

    for step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with precision_context(device, use_mixed_precision):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Compute the loss in full precision for numerical stability.
            logits = logits.float()
            loss = loss_function(logits, labels)

        # The scaler multiplies the loss before backpropagation so small
        # half-precision gradients do not round to zero.
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        # Clip gradients so one bad batch cannot blow up training.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss = total_loss + loss.item()

        if (step + 1) % 50 == 0:
            average_loss = total_loss / (step + 1)
            elapsed_minutes = (time.time() - start) / 60
            print("    epoch %d, step %d / %d, loss %.4f, %.1f min" % (
                epoch_number, step + 1, len(data_loader), average_loss, elapsed_minutes))

    return total_loss / len(data_loader)


def save_model(model, tokenizer, model_dir, model_name, stage, labels_in_order):
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    description = {
        "kind": "transformer",
        "stage": stage,
        "labels": labels_in_order,
        "base_model": model_name,
        "tokenizer_settings": tokenizer_settings(model_name),
    }
    save_json(description, os.path.join(model_dir, "triage_model.json"))


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.class_weighting is None:
        if args.stage == 1:
            args.class_weighting = "none"
        else:
            args.class_weighting = "sqrt_balanced"

    device, use_mixed_precision = choose_device()
    print("Using device: " + device.type)
    if device.type == "cpu":
        print("WARNING: no GPU found. Training on CPU will be very slow.")

    model_short = short_model_name(args.model_name)
    run_dir = run_directory(model_short, args.stage, args.seed)
    model_dir = os.path.join(run_dir, "model")
    print("Training " + args.model_name + " for Stage " + str(args.stage)
          + ", seed " + str(args.seed))
    print("Output folder: " + run_dir)

    # Data ------------------------------------------------------------------
    train_rows = load_processed_split(args.data_dir, args.stage, "train")
    dev_rows = load_processed_split(args.data_dir, args.stage, "dev")
    test_rows = load_processed_split(args.data_dir, args.stage, "test")

    if args.max_train_examples is not None:
        if args.max_train_examples < len(train_rows):
            sampler = random.Random(args.seed)
            train_rows = sampler.sample(train_rows, args.max_train_examples)
            print("DEBUG MODE: training on only " + str(len(train_rows)) + " tweets")

    label_set = set()
    for row in train_rows:
        label_set.add(row["label"])
    labels_in_order = sorted(label_set)

    label_to_id = {}
    id_to_label = {}
    for index in range(len(labels_in_order)):
        label_to_id[labels_in_order[index]] = index
        id_to_label[index] = labels_in_order[index]

    # Dev or test tweets with a label never seen in training cannot be
    # predicted correctly, so we stop rather than silently mis-score them.
    for rows, split_name in [(dev_rows, "dev"), (test_rows, "test")]:
        for row in rows:
            if row["label"] not in label_to_id:
                raise RuntimeError("The " + split_name + " set has label '" + row["label"]
                                   + "', which never appears in training.")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    print("Tokenizing train / dev / test...")
    train_dataset = build_dataset(train_rows, tokenizer, label_to_id, args.max_length)
    dev_dataset = build_dataset(dev_rows, tokenizer, label_to_id, args.max_length)
    test_dataset = build_dataset(test_rows, tokenizer, label_to_id, args.max_length)

    # The collator pads each batch to its longest tweet.
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size,
                            shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False, collate_fn=collator)

    # Model, loss, optimizer --------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels_in_order),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.to(device)

    training_labels = []
    for row in train_rows:
        training_labels.append(row["label"])
    weights = compute_class_weights(labels_in_order, training_labels, args.class_weighting)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    weight_tensor = weight_tensor.to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    print("Class weights (" + args.class_weighting + "):")
    for index in range(len(labels_in_order)):
        print("    %-40s %.3f" % (labels_in_order[index], weights[index]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # The gradient scaler is only needed for mixed precision on NVIDIA GPUs.
    # When it is disabled, its methods just pass values through unchanged.
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    # Training loop -----------------------------------------------------------
    dev_gold = []
    for row in dev_rows:
        dev_gold.append(row["label"])

    history = []
    best_dev_f1 = -1.0
    best_epoch = None
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        print("Epoch " + str(epoch) + " of " + str(args.epochs))
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                                     loss_function, device, use_mixed_precision, epoch)

        dev_probabilities = predict_probabilities(model, dev_loader, device,
                                                  use_mixed_precision)
        dev_predicted, dev_confidences = probabilities_to_predictions(
            dev_probabilities, labels_in_order)
        dev_metrics = compute_metrics(dev_gold, dev_predicted, labels_in_order)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_weighted_f1": dev_metrics["weighted_f1"],
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        history.append(epoch_record)
        print("  train loss %.4f, dev weighted F1 %.4f, dev macro F1 %.4f" % (
            train_loss, dev_metrics["weighted_f1"], dev_metrics["macro_f1"]))

        if dev_metrics["weighted_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["weighted_f1"]
            best_epoch = epoch
            save_model(model, tokenizer, model_dir, args.model_name, args.stage,
                       labels_in_order)
            print("  new best on dev, saved model")

    training_minutes = (time.time() - training_start) / 60

    # Test (once, with the best dev epoch) -----------------------------------
    print("Loading the best model (epoch " + str(best_epoch) + ") for testing...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    test_probabilities = predict_probabilities(model, test_loader, device,
                                               use_mixed_precision)
    test_predicted, test_confidences = probabilities_to_predictions(
        test_probabilities, labels_in_order)

    informative_probabilities = None
    if args.stage == 1:
        informative_index = label_to_id[config.INFORMATIVE_LABEL]
        informative_probabilities = []
        for probability_row in test_probabilities:
            informative_probabilities.append(float(probability_row[informative_index]))

    settings = {
        "base_model": args.model_name,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "class_weighting": args.class_weighting,
        "class_weights": weights,
        "train_size": len(train_rows),
        "debug_max_train_examples": args.max_train_examples,
        "mixed_precision": use_mixed_precision,
        "device": str(device),
        "training_minutes": training_minutes,
        "history": history,
    }
    save_run(run_dir, model_short, args.stage, args.seed, test_rows, test_predicted,
             test_confidences, informative_probabilities, labels_in_order, settings)


if __name__ == "__main__":
    main()
