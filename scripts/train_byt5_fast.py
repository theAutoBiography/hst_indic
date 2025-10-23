#!/usr/bin/env python3
"""
Fast ByT5 training with optimized data loading
Converts all data to IITHLP transliteration for cross-language compatibility
"""

import json
import torch
import os
from pathlib import Path
from typing import List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import numpy as np

# IITHLP transliteration for cross-language support
try:
    from iithlp import to_roman
    TRANSLITERATION_AVAILABLE = True
    print("✓ IITHLP transliteration available")
except ImportError:
    print("WARNING: iithlp not installed. Run: pip install iithlp")
    print("Training will use original Devanagari script")
    TRANSLITERATION_AVAILABLE = False

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def transliterate_to_iithlp(text: str) -> str:
    """Convert Devanagari text to IITHLP Roman script"""
    if not TRANSLITERATION_AVAILABLE:
        return text
    try:
        return to_roman(text)
    except Exception as e:
        print(f"Warning: Failed to transliterate '{text}': {e}")
        return text


def load_and_prepare_data(tokenizer, max_examples=None):
    """Load and prepare all data at once - faster than lazy loading"""

    print("Loading DCS data...")
    with open("data/processed/dcs_sandhi_splits.json", 'r') as f:
        dcs_data = json.load(f)

    print("Loading sandhiset data...")
    sandhiset_examples = []
    try:
        with open("data/annotations/sandhiset.txt", 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '=>' not in line:
                    continue
                parts = line.split('=>')
                if len(parts) == 2:
                    surface = parts[0].strip()
                    split_words = [w.strip() for w in parts[1].split('+')]
                    sandhiset_examples.append((surface, ' + '.join(split_words)))
    except:
        print("Sandhiset not found, using only DCS")

    # Combine examples
    all_examples = []
    for item in dcs_data:
        surface = item['surface_form']
        split = ' + '.join(item['split_words'])
        all_examples.append((surface, split))

    all_examples.extend(sandhiset_examples)

    if max_examples:
        all_examples = all_examples[:max_examples]

    print(f"Total examples (Devanagari): {len(all_examples):,}")

    # Convert to IITHLP transliteration
    if TRANSLITERATION_AVAILABLE:
        print("Converting to IITHLP Roman script...")
        all_examples = [
            (transliterate_to_iithlp(surface), transliterate_to_iithlp(split))
            for surface, split in all_examples
        ]
        print("✓ Transliteration complete")

    print(f"Total examples: {len(all_examples):,}")

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_examples)

    # Split
    n_train = int(0.85 * len(all_examples))
    n_val = int(0.10 * len(all_examples))

    train_examples = all_examples[:n_train]
    val_examples = all_examples[n_train:n_train + n_val]
    test_examples = all_examples[n_train + n_val:]

    print(f"Train: {len(train_examples):,}")
    print(f"Val: {len(val_examples):,}")
    print(f"Test: {len(test_examples):,}")

    # Tokenize all at once - MUCH faster
    print("Tokenizing all data...")

    def tokenize_batch(examples):
        inputs = [f"split: {surface}" for surface, _ in examples]
        targets = [split for _, split in examples]

        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=False)
        labels = tokenizer(targets, max_length=128, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]

        return Dataset.from_dict(model_inputs)

    train_dataset = tokenize_batch(train_examples)
    val_dataset = tokenize_batch(val_examples)
    test_dataset = tokenize_batch(test_examples)

    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_preds, tokenizer):
    """Compute exact match accuracy"""
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact_matches = sum(
        pred.strip() == label.strip()
        for pred, label in zip(decoded_preds, decoded_labels)
    )

    return {
        "exact_match_accuracy": exact_matches / len(decoded_preds) * 100,
        "exact_matches": exact_matches,
        "total": len(decoded_preds)
    }


def train_model():
    """Train ByT5 model - optimized version"""

    print("=" * 60)
    print("ByT5 SANDHI TRAINING (FAST)")
    print("=" * 60)
    print()

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free_mem = (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_allocated(0)) / 1e9
        print(f"Available: {free_mem:.1f} GB")
        print()

    # Load model
    model_name = "google/byt5-small"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # NO gradient checkpointing - it's making it slow!
    # model.gradient_checkpointing_enable()

    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(tokenizer)

    # Training arguments - OPTIMIZED
    output_path = Path("./models/byt5-sandhi-fast")
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=3,  # Reduced from 5
        per_device_train_batch_size=16,  # Increased from 8
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  # Simulate batch_size=32
        learning_rate=1e-4,  # FIXED: Lower LR for character-level model
        max_grad_norm=1.0,  # FIXED: Gradient clipping to prevent explosion
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,  # More frequent logging
        eval_strategy="steps",
        eval_steps=10000,
        save_strategy="steps",
        save_steps=10000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match_accuracy",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=128,
        bf16=torch.cuda.is_bf16_supported(),  # FIXED: Use bf16 if available (more stable)
        fp16=not torch.cuda.is_bf16_supported(),  # Fallback to fp16
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_pin_memory=True,  # Faster data transfer
        gradient_checkpointing=False,  # DISABLED - too slow
        optim="adamw_torch",  # Faster than Adafactor
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
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print("Optimizations:")
    print("  ✓ Batch size: 16 (2x faster)")
    print("  ✓ Parallel data loading: 4 workers")
    print("  ✓ Gradient checkpointing: DISABLED (was slowing down)")
    print("  ✓ Epochs: 3 (reduced for speed)")
    print("  ✓ Pre-tokenized data (no lazy loading)")
    print()
    print("Fixed hyperparameters:")
    print("  ✓ Learning rate: 1e-4 (was 5e-4 - too high!)")
    print("  ✓ Gradient clipping: 1.0 (prevents NaN)")
    print("  ✓ bf16/fp16: Auto-detect (more stable)")
    print()
    if TRANSLITERATION_AVAILABLE:
        print("  ✓ Using IITHLP transliteration (cross-language support)")
    print()
    print("Expected: ~2-3 hours (much faster!)")
    print()

    trainer.train()

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATING")
    print("=" * 60)

    test_results = trainer.predict(test_dataset)
    print("\nTest Results:")
    print(json.dumps(test_results.metrics, indent=2))

    # Save
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    with open(output_path / "test_results.json", 'w') as f:
        json.dump(test_results.metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model: {output_path}")
    print(f"Accuracy: {test_results.metrics['test_exact_match_accuracy']:.2f}%")
    print()


if __name__ == "__main__":
    train_model()
