#!/usr/bin/env python3
"""
Train ByT5 model with low memory settings
For shared GPU environments
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
from tqdm import tqdm

# Set environment variable for memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def load_dcs_data(max_examples: int = None) -> List[Tuple[str, str]]:
    """Load DCS sandhi splits"""
    data_path = Path("data/processed/dcs_sandhi_splits.json")

    if not data_path.exists():
        print(f"ERROR: {data_path} not found!")
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
    """Train ByT5 model with low memory settings"""

    print("=" * 60)
    print("ByT5 SANDHI TRAINING (LOW MEMORY MODE)")
    print("=" * 60)
    print()

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {gpu_mem:.1f} GB")

        # Check available memory
        torch.cuda.empty_cache()
        free_mem = (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_allocated(0)) / 1e9
        print(f"Available GPU Memory: {free_mem:.1f} GB")
        print()

        if free_mem < 10:
            print("⚠️  WARNING: Less than 10GB free. Other processes using GPU.")
            print("   Training will use minimal memory settings.")
            print()

    # Load data
    print("Loading training data...")
    dcs_examples = load_dcs_data()
    sandhiset_examples = load_sandhiset_data()
    all_examples = dcs_examples + sandhiset_examples

    print(f"\nTotal examples: {len(all_examples):,}")

    if not all_examples:
        print("ERROR: No training data found!")
        return

    # Load tokenizer and model
    model_name = "google/byt5-small"  # Use smaller model for low memory
    print(f"\nLoading model: {model_name} (smaller for memory constraints)")

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Prepare datasets
    print("\nPreparing datasets...")

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

    # Training arguments - LOW MEMORY settings
    output_path = Path("./models/byt5-sandhi-small")
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=5,
        per_device_train_batch_size=8,  # REDUCED from 32
        per_device_eval_batch_size=8,   # REDUCED from 32
        gradient_accumulation_steps=4,  # Simulate batch_size=32
        learning_rate=3e-4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(output_path / "logs"),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=1000,  # Less frequent eval to save time
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,  # Keep fewer checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="exact_match_accuracy",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=128,
        fp16=True,  # Mixed precision
        fp16_full_eval=True,  # FP16 for eval too
        gradient_checkpointing=True,  # Save memory
        optim="adafactor",  # Memory-efficient optimizer
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
    print("Settings:")
    print("  - Model: byt5-small (reduced size)")
    print("  - Batch size: 8 (with gradient accumulation)")
    print("  - Gradient checkpointing: enabled")
    print("  - FP16: enabled")
    print("  - Optimizer: Adafactor (memory efficient)")
    print()

    trainer.train()

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    test_results = trainer.predict(test_dataset)
    print("\nTest Results:")
    print(json.dumps(test_results.metrics, indent=2))

    # Save
    print(f"\nSaving model to {output_path}")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    with open(output_path / "test_results.json", 'w') as f:
        json.dump(test_results.metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel: {output_path}")
    print(f"Test accuracy: {test_results.metrics['test_exact_match_accuracy']:.2f}%")
    print()


if __name__ == "__main__":
    train_model()
