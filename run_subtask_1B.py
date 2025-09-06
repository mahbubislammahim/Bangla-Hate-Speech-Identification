#!/usr/bin/env python3
"""
Hate Speech Identification Shared Task: Subtask 1B
Using Mahim47/banglabert-hatespeech-subtask1b pre-trained model

This script loads the pre-trained model and generates predictions for the test data.
"""

import logging
import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

def load_data():
    """Load the training, validation, and test data"""
    print("📂 Loading data files...")
    
    # Data file paths
    train_file = 'data/subtask_1B/blp25_hatespeech_subtask_1B_train.tsv'
    validation_file = 'data/subtask_1B/blp25_hatespeech_subtask_1B_dev.tsv'
    test_file = 'data/subtask_1B/blp25_hatespeech_subtask_1B_dev_test.tsv'
    
    # Label mapping for subtask 1B
    l2id = {'None': 0, 'Society': 1, 'Organization': 2, 'Community': 3, 'Individual': 4}
    id2l = {v: k for k, v in l2id.items()}
    
    # Load training data
    train_df = pd.read_csv(train_file, sep='\t')
    train_df['label'] = train_df['label'].map(l2id).fillna(0).astype(int)
    train_dataset = Dataset.from_pandas(train_df)
    
    # Load validation data
    validation_df = pd.read_csv(validation_file, sep='\t')
    validation_df['label'] = validation_df['label'].map(l2id).fillna(0).astype(int)
    validation_dataset = Dataset.from_pandas(validation_df)
    
    # Load test data (no labels)
    test_df = pd.read_csv(test_file, sep='\t')
    test_dataset = Dataset.from_pandas(test_df)
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(validation_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")
    
    return train_dataset, validation_dataset, test_dataset, l2id, id2l

def load_pretrained_model():
    """Load the pre-trained Mahim47/banglabert-hatespeech-subtask1b model"""
    print("🤖 Loading pre-trained model: Mahim47/banglabert-hatespeech-subtask1b")
    
    model_name = "Mahim47/banglabert-hatespeech-subtask1b"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Number of labels: {config.num_labels}")
    
    return model, tokenizer, config

def preprocess_data(datasets, tokenizer, max_seq_length=128):
    """Preprocess the datasets using the tokenizer"""
    print("🔧 Preprocessing data...")
    
    def preprocess_function(examples):
        # Tokenize the text
        result = tokenizer(
            examples['text'],
            padding='max_length',
            max_length=max_seq_length,
            truncation=True,
            return_tensors=None
        )
        
        # Add labels if they exist
        if 'label' in examples:
            result['label'] = examples['label']
        
        return result
    
    # Apply preprocessing to all datasets
    processed_datasets = {}
    for name, dataset in datasets.items():
        # Remove id column before preprocessing
        if 'id' in dataset.column_names:
            dataset = dataset.remove_columns("id")
        
        processed_datasets[name] = dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=True,
            desc=f"Preprocessing {name} dataset"
        )
    
    print("✅ Data preprocessing completed!")
    return processed_datasets

def setup_trainer(model, tokenizer, train_dataset, eval_dataset):
    """Setup the trainer for evaluation and prediction"""
    print("⚙️ Setting up trainer...")
    
    # Training arguments (minimal for evaluation/prediction)
    training_args = TrainingArguments(
        output_dir="./results_subtask_1B/",
        overwrite_output_dir=True,
        remove_unused_columns=False,
        per_device_eval_batch_size=8,
        dataloader_pin_memory=False,
        report_to=None,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    
    print("✅ Trainer setup completed!")
    return trainer

def evaluate_model(trainer, eval_dataset, id2l):
    """Evaluate the model on validation data"""
    print("🔍 Evaluating model on validation data...")
    
    # Get predictions
    eval_results = trainer.predict(eval_dataset, metric_key_prefix="eval")
    predictions = np.argmax(eval_results.predictions, axis=1)
    
    # Get true labels
    true_labels = eval_results.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='micro')
    
    print(f"📊 Validation Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1: {f1:.4f}")
    
    return accuracy, precision, recall, f1

def generate_predictions(trainer, test_dataset, id2l, output_file, original_test_dataset):
    """Generate predictions for test data"""
    print("🔮 Generating predictions for test data...")
    
    # Get predictions
    predictions = trainer.predict(test_dataset, metric_key_prefix="predict")
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    # Get test IDs from original dataset
    test_ids = original_test_dataset['id']
    
    # Save predictions in required format
    with open(output_file, "w", encoding='utf-8') as writer:
        writer.write("id\tlabel\tmodel\n")
        for index, (test_id, pred_label) in enumerate(zip(test_ids, pred_labels)):
            label_name = id2l[pred_label]
            writer.write(f"{test_id}\t{label_name}\tMahim47/banglabert-hatespeech-subtask1b\n")
    
    print(f"✅ Predictions saved to: {output_file}")
    print(f"📊 Generated {len(pred_labels)} predictions")

def run_official_evaluation(pred_file, gold_file):
    """Run official evaluation using the scorer"""
    print("🏆 Running official evaluation...")
    
    # Import scorer functions
    sys.path.append('.')
    from scorer.task import evaluate, _read_tsv_input_file, _read_gold_labels_file
    
    # Read predictions and gold labels
    pred_labels = _read_tsv_input_file(pred_file)
    gold_labels = _read_gold_labels_file(gold_file)
    
    # Run evaluation
    acc, precision, recall, f1 = evaluate(pred_labels, gold_labels, '1B')
    
    print(f"📊 OFFICIAL COMPETITION SCORES:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1: {f1:.4f}")
    
    return acc, precision, recall, f1

def main():
    """Main function to run the complete pipeline"""
    print("🚀 Starting Subtask 1B with Mahim47/banglabert-hatespeech-subtask1b model")
    print("=" * 60)
    
    # Load data
    train_dataset, validation_dataset, test_dataset, l2id, id2l = load_data()
    
    # Store original datasets for ID access later
    original_validation_dataset = validation_dataset
    original_test_dataset = test_dataset
    
    # Load pre-trained model
    model, tokenizer, config = load_pretrained_model()
    
    # Preprocess data
    datasets = {
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    }
    processed_datasets = preprocess_data(datasets, tokenizer)
    
    # Setup trainer
    trainer = setup_trainer(
        model, 
        tokenizer, 
        processed_datasets['train'], 
        processed_datasets['validation']
    )
    
    # Evaluate on validation data
    accuracy, precision, recall, f1 = evaluate_model(
        trainer, 
        processed_datasets['validation'], 
        id2l
    )
    
    # Generate predictions for test data
    output_file = "subtask_1B_mahim47_predictions.tsv"
    generate_predictions(
        trainer, 
        processed_datasets['test'], 
        id2l, 
        output_file,
        original_test_dataset
    )
    
    # Run official evaluation on validation data
    print("\n" + "=" * 60)
    print("🏆 OFFICIAL EVALUATION ON VALIDATION DATA")
    print("=" * 60)
    
    # Create validation predictions file for official evaluation
    val_pred_file = "subtask_1B_mahim47_validation_predictions.tsv"
    val_predictions = trainer.predict(processed_datasets['validation'], metric_key_prefix="eval")
    val_pred_labels = np.argmax(val_predictions.predictions, axis=1)
    
    # Save validation predictions
    with open(val_pred_file, "w", encoding='utf-8') as writer:
        writer.write("id\tlabel\tmodel\n")
        for index, (test_id, pred_label) in enumerate(zip(original_validation_dataset['id'], val_pred_labels)):
            label_name = id2l[pred_label]
            writer.write(f"{test_id}\t{label_name}\tMahim47/banglabert-hatespeech-subtask1b\n")
    
    # Run official evaluation
    gold_file = "data/subtask_1B/blp25_hatespeech_subtask_1B_dev.tsv"
    official_acc, official_precision, official_recall, official_f1 = run_official_evaluation(
        val_pred_file, 
        gold_file
    )
    
    print("\n" + "=" * 60)
    print("🎉 COMPLETED!")
    print("=" * 60)
    print(f"📁 Test predictions: {output_file}")
    print(f"📁 Validation predictions: {val_pred_file}")
    print(f"📊 Final Official F1 Score: {official_f1:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
