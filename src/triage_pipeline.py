"""
The two-stage triage pipeline, used by the demo server.

    tweet -> Stage 1: informative?
               no  -> "filtered out" (shown in its own tab, never deleted)
               yes -> Stage 2: which humanitarian category?

Each stage is loaded from a saved model folder (the model/ folder inside a
run). The folder's triage_model.json says whether it holds a transformer
(BERTweet, RoBERTa) or a TF-IDF model, so the same code serves either.
"""

import os

import config
from crisisbench_io import load_json


class StageModel:
    """One trained stage. predict() returns, for each text, the predicted
    label and a dictionary of probabilities for every label."""

    def __init__(self, model_dir, batch_size=32):
        description_path = os.path.join(model_dir, "triage_model.json")
        if not os.path.exists(description_path):
            raise FileNotFoundError(description_path + " not found. Is " + model_dir
                                    + " the model/ folder of a finished run?")

        self.description = load_json(description_path)
        self.kind = self.description["kind"]
        self.labels = self.description["labels"]
        self.batch_size = batch_size

        if self.kind == "transformer":
            self._load_transformer(model_dir)
        elif self.kind == "tfidf":
            self._load_tfidf(model_dir)
        else:
            raise ValueError("Unknown model kind: " + self.kind)

    def _load_transformer(self, model_dir):
        # Imported here so the TF-IDF backup works even without torch installed.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        tokenizer_settings = self.description["tokenizer_settings"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, **tokenizer_settings)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def _load_tfidf(self, model_dir):
        import joblib

        model_path = os.path.join(model_dir, self.description["model_file"])
        self.model = joblib.load(model_path)

    def predict(self, texts):
        if self.kind == "transformer":
            probability_rows = self._transformer_probabilities(texts)
        else:
            probability_rows = self.model.predict_proba(texts)

        results = []
        for probability_row in probability_rows:
            probabilities = {}
            best_label = None
            best_probability = -1.0
            for index in range(len(self.labels)):
                label = self.labels[index]
                probability = float(probability_row[index])
                probabilities[label] = probability
                if probability > best_probability:
                    best_probability = probability
                    best_label = label
            results.append({"label": best_label, "probabilities": probabilities})
        return results

    def _transformer_probabilities(self, texts):
        torch = self.torch
        all_rows = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            encoded = self.tokenizer(batch_texts, truncation=True, max_length=128,
                                     padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            probabilities = torch.softmax(outputs.logits, dim=-1)
            probabilities = probabilities.cpu()
            for row in probabilities.tolist():
                all_rows.append(row)
        return all_rows


class TriagePipeline:
    def __init__(self, stage1_dir, stage2_dir, threshold=config.DEFAULT_STAGE1_THRESHOLD):
        print("Loading Stage 1 from " + stage1_dir)
        self.stage1 = StageModel(stage1_dir)
        print("Loading Stage 2 from " + stage2_dir)
        self.stage2 = StageModel(stage2_dir)
        self.threshold = threshold

        if config.INFORMATIVE_LABEL not in self.stage1.labels:
            raise ValueError("The Stage 1 model does not have the label '"
                             + config.INFORMATIVE_LABEL + "'. Is it really a Stage 1 model?")

    def classify(self, texts):
        """Return one result per text:
            informative          True / False (Stage 1 decision)
            prob_informative     Stage 1 probability
            category             Stage 2 label, or None if filtered out
            category_confidence  Stage 2 probability of that label, or None
            end_to_end_label     what the pipeline outputs overall
        """
        stage1_results = self.stage1.predict(texts)

        # Only informative tweets go to Stage 2.
        informative_indexes = []
        informative_texts = []
        for index in range(len(texts)):
            probability = stage1_results[index]["probabilities"][config.INFORMATIVE_LABEL]
            if probability >= self.threshold:
                informative_indexes.append(index)
                informative_texts.append(texts[index])

        stage2_by_index = {}
        if len(informative_texts) > 0:
            stage2_results = self.stage2.predict(informative_texts)
            for position in range(len(informative_indexes)):
                stage2_by_index[informative_indexes[position]] = stage2_results[position]

        results = []
        for index in range(len(texts)):
            probability = stage1_results[index]["probabilities"][config.INFORMATIVE_LABEL]
            result = {"prob_informative": probability}

            if index in stage2_by_index:
                stage2_result = stage2_by_index[index]
                category = stage2_result["label"]
                result["informative"] = True
                result["category"] = category
                result["category_confidence"] = stage2_result["probabilities"][category]
                if category == config.NOT_HUMANITARIAN_LABEL:
                    result["end_to_end_label"] = config.NOT_ACTIONABLE_LABEL
                else:
                    result["end_to_end_label"] = category
            else:
                result["informative"] = False
                result["category"] = None
                result["category_confidence"] = None
                result["end_to_end_label"] = config.NOT_ACTIONABLE_LABEL

            results.append(result)
        return results
