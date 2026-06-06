"""
train_pipeline.py
-----------------
MLOps Assignment 2 — Book Genre Classification
IIT Jodhpur | PGD AI Program

Full pipeline:
  1. Download & sample Goodreads reviews
  2. Tokenize with DistilBERT
  3. Fine-tune using HuggingFace Trainer
  4. Track experiments with Weights & Biases
  5. Evaluate on test set, save classification report
  6. Push model + tokenizer to Hugging Face Hub

Usage:
    export WANDB_API_KEY=<your_key>
    export HF_TOKEN=<your_token>
    python train_pipeline.py

On Kaggle: load keys via kaggle_secrets.UserSecretsClient instead of env vars.
GPU strongly recommended. For CPU-only, set DEVICE="cpu" and SAMPLES_PER_GENRE=200.
"""

import os
import json
import random
import gzip
import requests
import numpy as np
import wandb

from io import BytesIO
from torch.utils.data import Dataset
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from huggingface_hub import login

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-cased"
MAX_TOKEN_LEN = 512
SAMPLES_PER_GENRE = 2000   # reduce to 200 for CPU runs
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
EPOCHS = 3
TRAIN_BATCH = 16
EVAL_BATCH = 32
LEARNING_RATE = 3e-5
WARMUP = 100
DECAY = 0.01
OUTPUT_DIR = "./checkpoints"
HF_REPO_NAME = "DishaSinghania/distilbert-goodreads-genres"
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN_NAME = "distilbert-goodreads-run"

# Genre names and their UCSD dataset URLs
GENRE_URLS = {
    "Children":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "Comics_Graphic":          "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "Fantasy_Paranormal":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "History_Biography":       "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "Mystery_Thriller_Crime":  "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "Poetry":                  "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "Romance":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "Young_Adult":             "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}

label2id = {genre: idx for idx, genre in enumerate(GENRE_URLS.keys())}
id2label = {idx: genre for genre, idx in label2id.items()}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# STEP 1: DATA LOADING
# ─────────────────────────────────────────────

def stream_reviews(url: str, max_samples: int) -> list[str]:
    """Stream a .json.gz file from URL, return up to max_samples review texts."""
    collected = []
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    buffer = BytesIO()
    for chunk in response.iter_content(chunk_size=65536):
        buffer.write(chunk)
    buffer.seek(0)

    with gzip.open(buffer, "rt", encoding="utf-8") as gz:
        for line in gz:
            if len(collected) >= max_samples * 5:   # over-sample then shuffle-pick
                break
            try:
                record = json.loads(line)
                text = record.get("review_text", "").strip()
                if len(text) > 20:
                    collected.append(text)
            except json.JSONDecodeError:
                continue

    random.shuffle(collected)
    return collected[:max_samples]


def build_dataset(samples_per_genre: int = SAMPLES_PER_GENRE):
    """Download all genres, return (texts, labels) lists."""
    all_texts, all_labels = [], []
    for genre, url in GENRE_URLS.items():
        print(f"  Downloading {genre}...")
        reviews = stream_reviews(url, samples_per_genre)
        all_texts.extend(reviews)
        all_labels.extend([label2id[genre]] * len(reviews))
        print(f"    → {len(reviews)} reviews loaded")
    return all_texts, all_labels


def train_test_split_by_label(texts, labels, train_ratio=TRAIN_RATIO):
    """Stratified split: keeps genre distribution equal in train/test."""
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre_id in range(len(GENRE_URLS)):
        indices = [i for i, l in enumerate(labels) if l == genre_id]
        random.shuffle(indices)
        cut = int(len(indices) * train_ratio)
        for i in indices[:cut]:
            train_texts.append(texts[i])
            train_labels.append(labels[i])
        for i in indices[cut:]:
            test_texts.append(texts[i])
            test_labels.append(labels[i])

    return train_texts, train_labels, test_texts, test_labels


# ─────────────────────────────────────────────
# STEP 2: TORCH DATASET
# ─────────────────────────────────────────────

class ReviewDataset(Dataset):
    def __init__(self, encodings: dict, label_list: list):
        self.encodings = encodings
        self.labels = label_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ─────────────────────────────────────────────
# STEP 3: METRICS
# ─────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, true_labels = eval_pred
    predicted = np.argmax(logits, axis=-1)
    acc = accuracy_score(true_labels, predicted)
    f1 = f1_score(true_labels, predicted, average="weighted")
    return {"accuracy": acc, "f1": f1}


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    # ── Auth ──
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
    if hf_token:
        login(token=hf_token)

    # ── W&B init ──
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH,
            "learning_rate": LEARNING_RATE,
            "max_token_length": MAX_TOKEN_LEN,
            "samples_per_genre": SAMPLES_PER_GENRE,
            "dataset": "UCSD Goodreads",
            "platform": "Kaggle T4 GPU",
        },
    )

    # ── Data ──
    print("\n[1/5] Loading dataset...")
    texts, labels = build_dataset()
    tr_texts, tr_labels, te_texts, te_labels = train_test_split_by_label(texts, labels)
    print(f"  Train: {len(tr_texts)} | Test: {len(te_texts)}")

    # ── Tokenizer ──
    print("\n[2/5] Tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(tr_texts, truncation=True, padding=True, max_length=MAX_TOKEN_LEN)
    test_enc  = tokenizer(te_texts, truncation=True, padding=True, max_length=MAX_TOKEN_LEN)

    train_dataset = ReviewDataset(train_enc, tr_labels)
    test_dataset  = ReviewDataset(test_enc,  te_labels)

    # ── Model ──
    print("\n[3/5] Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    # ── Training ──
    print("\n[4/5] Training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        warmup_steps=WARMUP,
        weight_decay=DECAY,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # ── Evaluation ──
    print("\n[5/5] Evaluating...")
    eval_results = trainer.evaluate()
    print(eval_results)

    wandb.log({
        "final/loss":     eval_results["eval_loss"],
        "final/accuracy": eval_results["eval_accuracy"],
        "final/f1":       eval_results["eval_f1"],
    })

    # Full classification report
    predictions = trainer.predict(test_dataset).predictions.argmax(-1)
    true_labels_list = [sample["labels"].item() for sample in test_dataset]
    genre_names = [id2label[i] for i in range(len(id2label))]

    report_dict = classification_report(
        true_labels_list,
        predictions,
        target_names=genre_names,
        output_dict=True,
    )
    print(classification_report(true_labels_list, predictions, target_names=genre_names))

    with open("eval_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # Upload report as W&B Artifact
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file("eval_report.json")
    wandb.log_artifact(artifact)

    # ── Push to HF Hub ──
    print("\nPushing to Hugging Face Hub...")
    model.push_to_hub(HF_REPO_NAME)
    tokenizer.push_to_hub(HF_REPO_NAME)

    wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{HF_REPO_NAME}"
    wandb.finish()
    print(f"\nDone! Model live at: https://huggingface.co/{HF_REPO_NAME}")


if __name__ == "__main__":
    main()
