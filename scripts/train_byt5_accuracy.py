#!/usr/bin/env python3
"""
ByT5-Sanskrit training for sandhi splitting - ACCURACY OPTIMIZED
Uses buddhist-nlp/byt5-sanskrit model pretrained on Sanskrit Digital Corpus
Converts all data to IAST transliteration (International Alphabet of Sanskrit Transliteration)

This version prioritizes accuracy over speed:
- More epochs (5 vs 3)
- Smaller batch size for stability (8 vs 16)
- Lower learning rate for careful optimization (5e-5 vs 1e-4)
- More frequent evaluation and checkpointing (5k vs 10k)
- Beam search for better generation quality
- Cosine LR schedule for better late-stage learning
- Full validation set (no reduction)

Expected training time: ~20-25 hours
Expected accuracy: 85-90%+
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

# IAST transliteration for Sanskrit (byt5-sanskrit is trained on IAST)
try:
    from indic_transliteration import sanscript
    TRANSLITERATION_AVAILABLE = True
    print("✓ IAST transliteration available (indic_transliteration)")
except ImportError:
    print("WARNING: indic_transliteration not installed. Run: pip install indic-transliteration")
    print("Training will use original Devanagari script")
    TRANSLITERATION_AVAILABLE = False

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def transliterate_to_iast(text: str) -> str:
    """Convert Devanagari text to IAST Roman script"""
    if not TRANSLITERATION_AVAILABLE:
        return text
    try:
        # Convert from Devanagari to IAST
        return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
    except Exception as e:
        print(f"Warning: Failed to transliterate '{text}': {e}")
        return text


def load_and_prepare_data(tokenizer, max_examples=None):
    """Load and prepare all data at once"""

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

    # Convert to IAST transliteration
    if TRANSLITERATION_AVAILABLE:
        print("Converting to IAST Roman script...")
        all_examples = [
            (transliterate_to_iast(surface), transliterate_to_iast(split))
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

    # ACCURACY MODE: Use full validation set (no reduction)
    print(f"Train: {len(train_examples):,}")
    print(f"Val: {len(val_examples):,} (full set for accurate evaluation)")
    print(f"Test: {len(test_examples):,}")

    # Tokenize all at once
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

    # Clip predictions to valid Unicode range to avoid chr() errors
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


def train_model():
    """Train ByT5 model - accuracy optimized version"""

    print("=" * 60)
    print("ByT5 SANDHI TRAINING (ACCURACY OPTIMIZED)")
    print("=" * 60)
    print()

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free_mem = (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_allocated(0)) / 1e9
        print(f"Available: {free_mem:.1f} GB")
        print()

    # Load model - using Sanskrit-pretrained ByT5
    model_name = "buddhist-nlp/byt5-sanskrit"
    print(f"Loading model: {model_name}")
    print("  (Pretrained on Sanskrit Digital Corpus with IAST transliteration)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(tokenizer)

    # Training arguments - ACCURACY OPTIMIZED
    output_path = Path("./models/byt5-sandhi-accuracy")
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),

        # ACCURACY: More epochs for better convergence
        num_train_epochs=5,

        # ACCURACY: Smaller batch size for more stable gradients
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        # ACCURACY: More gradient accumulation (effective batch=32)
        gradient_accumulation_steps=4,

        # ACCURACY: Lower learning rate for careful optimization
        learning_rate=5e-5,
        max_grad_norm=1.0,

        # ACCURACY: More warmup for smooth start
        warmup_steps=1000,
        warmup_ratio=0.1,
        weight_decay=0.01,

        # ACCURACY: More frequent logging for monitoring
        logging_steps=100,

        # ACCURACY: More frequent evaluation for better model selection
        eval_strategy="steps",
        eval_steps=5000,

        # ACCURACY: More frequent saves and keep more checkpoints
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=3,

        load_best_model_at_end=True,
        metric_for_best_model="exact_match_accuracy",
        greater_is_better=True,

        # ACCURACY: Enable beam search for better generation
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,

        # ACCURACY: Use bf16/fp16 for stability
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        # ACCURACY: Fewer workers for stability
        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        gradient_checkpointing=False,

        # ACCURACY: Cosine schedule for better late-stage learning
        optim="adamw_torch",
        lr_scheduler_type="cosine",

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
    print("Accuracy optimizations:")
    print("  ✓ Epochs: 5 (vs 3 in fast mode)")
    print("  ✓ Batch size: 8 (smaller for stability)")
    print("  ✓ Gradient accumulation: 4 (effective batch=32)")
    print("  ✓ Learning rate: 5e-5 (lower for careful optimization)")
    print("  ✓ Warmup: 1000 steps (more gradual start)")
    print("  ✓ Eval frequency: 5000 steps (more frequent)")
    print("  ✓ Beam search: 4 beams (better generation)")
    print("  ✓ LR schedule: Cosine (better late-stage learning)")
    print("  ✓ Full validation set (no reduction)")
    print()
    print("Fixed hyperparameters:")
    print("  ✓ Gradient clipping: 1.0 (prevents NaN)")
    print("  ✓ bf16/fp16: Auto-detect (more stable)")
    print()
    if TRANSLITERATION_AVAILABLE:
        print("  ✓ Using IAST transliteration (optimized for byt5-sanskrit)")
    print()
    print("Expected: ~20-25 hours (optimized for accuracy!)")
    print("Target accuracy: 85-90%+")
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
