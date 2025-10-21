#!/usr/bin/env python3
"""
Train ByT5 model for Sanskrit sandhi splitting
Uses 742k examples: 625k DCS + 117k sandhiset
"""

import json
import torch
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import numpy as np
from tqdm import tqdm


def load_dcs_data(max_examples: int = None) -> List[Tuple[str, str]]:
    """Load DCS sandhi splits"""
    data_path = Path("data/processed/dcs_sandhi_splits.json")

    if not data_path.exists():
        print(f"ERROR: {data_path} not found!")
        print("Run: python scripts/extract_from_dcs_conllu.py")
        return []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data:
        surface = item['surface_form']
        split = ' + '.join(item['split_words'])
        examples.append((surface, split))

    if max_examples:
        examples = examples[:max_examples]

    print(f"Loaded {len(examples):,} DCS examples")
    return examples


def load_sandhiset_data(max_examples: int = None) -> List[Tuple[str, str]]:
    """Load sandhiset manual annotations"""
    data_path = Path("data/annotations/sandhiset.txt")

    if not data_path.exists():
        print(f"WARNING: {data_path} not found, skipping sandhiset")
        return []

    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '=>' not in line:
                continue

            parts = line.split('=>')
            if len(parts) != 2:
                continue

            surface = parts[0].strip()
            split_words = [w.strip() for w in parts[1].split('+')]
            split = ' + '.join(split_words)
            examples.append((surface, split))

    if max_examples:
        examples = examples[:max_examples]

    print(f"Loaded {len(examples):,} sandhiset examples")
    return examples


def prepare_dataset(examples: List[Tuple[str, str]], tokenizer):
    """Convert examples to model inputs"""
    inputs = []
    targets = []

    for surface, split in tqdm(examples, desc="Preparing dataset"):
        # Input: "split: <compound>"
        input_text = f"split: {surface}"
        inputs.append(input_text)
        targets.append(split)

    # Tokenize
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=False)
    labels = tokenizer(targets, max_length=128, truncation=True, padding=False)

    model_inputs["labels"] = labels["input_ids"]

    return Dataset.from_dict(model_inputs)


def compute_metrics(eval_preds):
    """Compute exact match accuracy"""
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute exact match
    exact_matches = sum(
        pred.strip() == label.strip()
        for pred, label in zip(decoded_preds, decoded_labels)
    )

    return {
        "exact_match_accuracy": exact_matches / len(decoded_preds) * 100,
        "exact_matches": exact_matches,
        "total": len(decoded_preds)
    }


def train_model(
    model_name: str = "google/byt5-base",
    output_dir: str = "./models/byt5-sandhi",
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    max_examples: int = None,
    use_fp16: bool = True,
    gradient_checkpointing: bool = False
):
    """Train ByT5 model on sandhi data"""

    print("=" * 60)
    print("ByT5 SANDHI TRAINING")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"FP16: {use_fp16}")
    print()

    # Load data
    print("Loading training data...")
    dcs_examples = load_dcs_data(max_examples)
    sandhiset_examples = load_sandhiset_data(max_examples)
    all_examples = dcs_examples + sandhiset_examples

    print(f"\nTotal examples: {len(all_examples):,}")

    if not all_examples:
        print("ERROR: No training data found!")
        return

    # Load tokenizer and model
    print(f"\nLoading model: {model_name}")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare datasets
    print("\nPreparing datasets...")

    # Shuffle and split: 85% train, 10% val, 5% test
    import random
    random.shuffle(all_examples)

    n_train = int(0.85 * len(all_examples))
    n_val = int(0.10 * len(all_examples))

    train_examples = all_examples[:n_train]
    val_examples = all_examples[n_train:n_train + n_val]
    test_examples = all_examples[n_train + n_val:]

    print(f"Train: {len(train_examples):,}")
    print(f"Val: {len(val_examples):,}")
    print(f"Test: {len(test_examples):,}")

    train_dataset = prepare_dataset(train_examples, tokenizer)
    val_dataset = prepare_dataset(val_examples, tokenizer)
    test_dataset = prepare_dataset(test_examples, tokenizer)

    # Training arguments
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=str(output_path / "logs"),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match_accuracy",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=128,
        fp16=use_fp16 and torch.cuda.is_available(),
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print()

    trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    test_results = trainer.predict(test_dataset)
    print("\nTest Results:")
    print(json.dumps(test_results.metrics, indent=2))

    # Save model
    print(f"\nSaving model to {output_path}")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save test results
    with open(output_path / "test_results.json", 'w') as f:
        json.dump(test_results.metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {output_path}")
    print(f"Test accuracy: {test_results.metrics['test_exact_match_accuracy']:.2f}%")
    print()


if __name__ == "__main__":
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()
    else:
        print("WARNING: No GPU detected. Training will be very slow!")
        print()

    # Train model
    train_model(
        model_name="google/byt5-base",  # 580M params
        output_dir="./models/byt5-sandhi",
        num_epochs=5,
        batch_size=32,  # Increase to 64 on A100 80GB
        learning_rate=3e-4,
        warmup_steps=500,
        max_examples=None,  # Use all data
        use_fp16=True,  # 2x faster on A100
        gradient_checkpointing=False,  # Enable if OOM
    )
