#!/usr/bin/env python3
"""
Train ByT5 model on a specific doping experiment dataset
Modified from train_byt5_fast.py to use experiment data
"""

import json
import torch
import os
import sys
from pathlib import Path
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
    print("WARNING: iithlp not installed")
    print("Training will use original scripts (Devanagari/Malayalam/Telugu)")
    TRANSLITERATION_AVAILABLE = False

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def transliterate_to_iithlp(text: str) -> str:
    """Convert text to IITHLP Roman script"""
    if not TRANSLITERATION_AVAILABLE:
        return text
    try:
        return to_roman(text)
    except Exception as e:
        return text


def load_experiment_data(exp_dir: str, tokenizer):
    """Load experiment data (train/val/test) and tokenize"""

    exp_path = Path(exp_dir)

    # Load JSON files
    with open(exp_path / "train.json", 'r') as f:
        train_data = json.load(f)
    with open(exp_path / "val.json", 'r') as f:
        val_data = json.load(f)
    with open(exp_path / "test.json", 'r') as f:
        test_data = json.load(f)

    print(f"Loaded experiment: {exp_path.name}")
    print(f"  Train: {len(train_data):,}")
    print(f"  Val: {len(val_data):,}")
    print(f"  Test: {len(test_data):,}")

    # Convert to (surface, split) tuples
    def prepare_examples(data):
        examples = []
        for item in data:
            surface = item['surface_form']
            split = ' + '.join(item['split_words'])

            # Transliterate if available
            if TRANSLITERATION_AVAILABLE:
                surface = transliterate_to_iithlp(surface)
                split = transliterate_to_iithlp(split)

            examples.append((surface, split))
        return examples

    train_examples = prepare_examples(train_data)
    val_examples = prepare_examples(val_data)
    test_examples = prepare_examples(test_data)

    # Limit validation set to 1000 examples for faster evaluation
    if len(val_examples) > 1000:
        print(f"  Reducing validation from {len(val_examples):,} to 1,000 for speed")
        val_examples = val_examples[:1000]

    # Tokenize
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

    # Clip predictions to valid Unicode range to avoid chr() errors
    # Valid Unicode range: 0 to 0x10FFFF (1,114,111)
    max_valid_id = tokenizer.vocab_size - 1
    predictions = np.clip(predictions, 0, max_valid_id)

    # Decode with error handling
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    except (ValueError, OverflowError) as e:
        print(f"Warning: Decoding error, using safer method: {e}")
        # Fallback: decode one by one, skip invalid
        decoded_preds = []
        for pred_ids in predictions:
            try:
                decoded_preds.append(tokenizer.decode(pred_ids, skip_special_tokens=True))
            except:
                decoded_preds.append("")  # Empty string for failed decodes

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


def train_experiment(exp_dir: str, output_suffix: str = ""):
    """Train model on experiment data"""

    print("=" * 60)
    print("BYT5 DOPING EXPERIMENT TRAINING")
    print("=" * 60)
    print(f"Experiment: {exp_dir}")
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

    # Load experiment data
    train_dataset, val_dataset, test_dataset = load_experiment_data(exp_dir, tokenizer)

    # Output directory
    exp_name = Path(exp_dir).name
    output_path = Path(f"./models/byt5_{exp_name}{output_suffix}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Training arguments - OPTIMIZED
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,  # FIXED: Lower LR for character-level model
        max_grad_norm=1.0,  # FIXED: Gradient clipping
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
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
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        optim="adamw_torch",
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
    if TRANSLITERATION_AVAILABLE:
        print("  ✓ Using IITHLP transliteration")
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

    # Save experiment info
    with open(output_path / "experiment_info.json", 'w') as f:
        json.dump({
            'experiment_dir': exp_dir,
            'experiment_name': exp_name,
            'test_accuracy': test_results.metrics['test_exact_match_accuracy']
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model: {output_path}")
    print(f"Accuracy: {test_results.metrics['test_exact_match_accuracy']:.2f}%")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_doping_experiment.py <experiment_dir> [output_suffix]")
        print("\nExample:")
        print("  python train_doping_experiment.py data/experiments/doping/malayalam_doping_05pct")
        sys.exit(1)

    exp_dir = sys.argv[1]
    output_suffix = sys.argv[2] if len(sys.argv) > 2 else ""

    train_experiment(exp_dir, output_suffix)
