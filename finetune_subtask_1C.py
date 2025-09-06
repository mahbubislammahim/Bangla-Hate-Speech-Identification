# Hate Speech Identification Shared Task: Subtask 1C at BLP Workshop @IJCNLP-AACL 2025
# Multi-label Classification: hate_type, hate_severity, and to_whom
# Modified version using BanglaBERT Large

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================

#!pip install transformers datasets evaluate accelerate huggingface_hub sentencepiece git+https://github.com/csebuetnlp/normalizer

# ============================================================================
# CELL 2: Import Libraries
# ============================================================================

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import datasets
import evaluate
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from normalizer import normalize

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ============================================================================
# CELL 3: Disable wandb
# ============================================================================

import os
os.environ["WANDB_DISABLED"] = "true"

# ============================================================================
# CELL 4: Define Data Files
# ============================================================================

train_file = 'blp25_hatespeech_subtask_1C_train.tsv'
validation_file = 'blp25_hatespeech_subtask_1C_dev.tsv'
test_file = 'blp25_hatespeech_subtask_1C_dev_test.tsv'

# ============================================================================
# CELL 5: Training Arguments (Optimized for BanglaBERT Large)
# ============================================================================

training_args = TrainingArguments(
    learning_rate=4e-5,
    num_train_epochs=2,  
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    output_dir="./banglabert_large_1C/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    local_rank=-1,
    load_best_model_at_end=True,
    resume_from_checkpoint=True,
    save_total_limit=2,
    save_strategy="epoch",  # Save every epoch
    eval_strategy="epoch",  # Evaluate every epoch
    metric_for_best_model="eval_average_f1",
    greater_is_better=True,
    warmup_ratio=0.08, 
    weight_decay=0.01, 
    logging_steps=50,
    report_to=None,
    dataloader_num_workers=0,
    fp16=True,
    lr_scheduler_type="linear", 
    save_steps=500,
    eval_steps=500,
)

max_train_samples = None
max_eval_samples = None
max_predict_samples = None
max_seq_length = 256 
batch_size = 16

# ============================================================================
# CELL 6: Setup Logging
# ============================================================================

transformers.utils.logging.set_verbosity_info()
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# ============================================================================
# CELL 7: Define Model (BANGLABERT LARGE)
# ============================================================================

model_name = 'csebuetnlp/banglabert'  
print(f"🤖 Using BanglaBERT Large model: {model_name}")

# ============================================================================
# CELL 8: Set Random Seed
# ============================================================================

set_seed(training_args.seed)

# ============================================================================
# CELL 9: Load Data Files and Define Label Mappings
# ============================================================================

# Label mappings for multi-label classification (based on actual data analysis)
hate_type_l2id = {
    'None': 0,           # For NaN/null values
    'Abusive': 1, 
    'Political Hate': 2, 
    'Profane': 3,
    'Religious Hate': 4, 
    'Sexism': 5
}
hate_severity_l2id = {'Little to None': 0, 'Mild': 1, 'Severe': 2}
to_whom_l2id = {
    'None': 0,           
    'Individual': 1, 
    'Organization': 2, 
    'Community': 3, 
    'Society': 4
}

# Reverse mappings
hate_type_id2l = {v: k for k, v in hate_type_l2id.items()}
hate_severity_id2l = {v: k for k, v in hate_severity_l2id.items()}
to_whom_id2l = {v: k for k, v in to_whom_l2id.items()}

print(f"Hate Type Labels: {hate_type_l2id}")
print(f"Hate Severity Labels: {hate_severity_l2id}")
print(f"To Whom Labels: {to_whom_l2id}")

# Load training data
train_df = pd.read_csv(train_file, sep='\t')
train_df['hate_type'] = train_df['hate_type'].map(hate_type_l2id).fillna(0).astype(int)
train_df['hate_severity'] = train_df['hate_severity'].map(hate_severity_l2id).fillna(0).astype(int)
train_df['to_whom'] = train_df['to_whom'].map(to_whom_l2id).fillna(0).astype(int)
train_df = Dataset.from_pandas(train_df)

# Load validation data
validation_df = pd.read_csv(validation_file, sep='\t')
validation_df['hate_type'] = validation_df['hate_type'].map(hate_type_l2id).fillna(0).astype(int)
validation_df['hate_severity'] = validation_df['hate_severity'].map(hate_severity_l2id).fillna(0).astype(int)
validation_df['to_whom'] = validation_df['to_whom'].map(to_whom_l2id).fillna(0).astype(int)
validation_df = Dataset.from_pandas(validation_df)

# Load test data
test_df = pd.read_csv(test_file, sep='\t')
test_df = Dataset.from_pandas(test_df)

data_files = {"train": train_df, "validation": validation_df, "test": test_df}
for key in data_files.keys():
    logger.info(f"loading a local file for {key}")

raw_datasets = DatasetDict(
    {"train": train_df, "validation": validation_df, "test": test_df}
)

print(f"✅ Training samples: {len(train_df)}")
print(f"✅ Validation samples: {len(validation_df)}")
print(f"✅ Test samples: {len(test_df)}")

# ============================================================================
# CELL 10: Check Label Distribution
# ============================================================================

# Show label distribution for each task
print("\n📊 Label distribution in training data:")
print("Hate Type:")
hate_type_counts = pd.read_csv(train_file, sep='\t')['hate_type'].value_counts()
for label, count in hate_type_counts.items():
    print(f"  {label}: {count}")

print("\nHate Severity:")
hate_severity_counts = pd.read_csv(train_file, sep='\t')['hate_severity'].value_counts()
for label, count in hate_severity_counts.items():
    print(f"  {label}: {count}")

print("\nTo Whom:")
to_whom_counts = pd.read_csv(train_file, sep='\t')['to_whom'].value_counts()
for label, count in to_whom_counts.items():
    print(f"  {label}: {count}")

# ============================================================================
# CELL 11: Custom Multi-Label Model Class
# ============================================================================

from torch import nn
from transformers import BertPreTrainedModel, BertModel

class BanglaBERTForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_hate_type_labels = len(hate_type_l2id)
        self.num_hate_severity_labels = len(hate_severity_l2id)
        self.num_to_whom_labels = len(to_whom_l2id)
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Separate classifiers for each task
        self.hate_type_classifier = nn.Linear(config.hidden_size, self.num_hate_type_labels)
        self.hate_severity_classifier = nn.Linear(config.hidden_size, self.num_hate_severity_labels)
        self.to_whom_classifier = nn.Linear(config.hidden_size, self.num_to_whom_labels)
        
        self.init_weights()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                hate_type_labels=None, hate_severity_labels=None, to_whom_labels=None, **kwargs):
        
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        hate_type_logits = self.hate_type_classifier(pooled_output)
        hate_severity_logits = self.hate_severity_classifier(pooled_output)
        to_whom_logits = self.to_whom_classifier(pooled_output)
        
        loss = None
        if hate_type_labels is not None and hate_severity_labels is not None and to_whom_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            hate_type_loss = loss_fct(hate_type_logits.view(-1, self.num_hate_type_labels), hate_type_labels.view(-1))
            hate_severity_loss = loss_fct(hate_severity_logits.view(-1, self.num_hate_severity_labels), hate_severity_labels.view(-1))
            to_whom_loss = loss_fct(to_whom_logits.view(-1, self.num_to_whom_labels), to_whom_labels.view(-1))
            loss = hate_type_loss + hate_severity_loss + to_whom_loss
        
        return {
            'loss': loss,
            'hate_type_logits': hate_type_logits,
            'hate_severity_logits': hate_severity_logits,
            'to_whom_logits': to_whom_logits
        }

# ============================================================================
# CELL 12: Load Model, Tokenizer, and Config
# ============================================================================

config = AutoConfig.from_pretrained(
    model_name,
    cache_dir=None,
    revision="main",
    use_auth_token=None,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=None,
    use_fast=True,
    revision="main",
    use_auth_token=None,
)

# Load our custom multi-label model
model = BanglaBERTForMultiLabelClassification.from_pretrained(
    model_name,
    config=config,
    cache_dir=None,
    revision="main",
    use_auth_token=None,
    ignore_mismatched_sizes=True,
)

print(f"✅ BanglaBERT Large model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# CELL 13: Preprocess Data
# ============================================================================

non_label_column_names = [name for name in raw_datasets["train"].column_names 
                         if name not in ["hate_type", "hate_severity", "to_whom"]]
# Prefer 'text' if available; otherwise fall back to first non-label column
sentence1_key = "text" if "text" in raw_datasets["train"].column_names else non_label_column_names[0]

padding = "max_length"

if max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.")
max_seq_length = min(max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    texts = examples[sentence1_key]
    texts = [normalize(str(t)) for t in texts]
    result = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True)
    
    # Add labels for training/validation
    if "hate_type" in examples:
        result["hate_type_labels"] = examples["hate_type"]
        result["hate_severity_labels"] = examples["hate_severity"]
        result["to_whom_labels"] = examples["to_whom"]
    
    return result

raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

# ============================================================================
# CELL 14: Prepare Datasets
# ============================================================================

if "train" not in raw_datasets:
    raise ValueError("requires a train dataset")
train_dataset = raw_datasets["train"]
if max_train_samples is not None:
    max_train_samples_n = min(len(train_dataset), max_train_samples)
    train_dataset = train_dataset.select(range(max_train_samples_n))

if "validation" not in raw_datasets:
    raise ValueError("requires a validation dataset")
eval_dataset = raw_datasets["validation"]
if max_eval_samples is not None:
    max_eval_samples_n = min(len(eval_dataset), max_eval_samples)
    eval_dataset = eval_dataset.select(range(max_eval_samples_n))

if "test" not in raw_datasets:
    raise ValueError("requires a test dataset")
predict_dataset = raw_datasets["test"]
if max_predict_samples is not None:
    max_predict_samples_n = min(len(predict_dataset), max_predict_samples)
    predict_dataset = predict_dataset.select(range(max_predict_samples_n))

# ============================================================================
# CELL 15: Log Sample Data
# ============================================================================

for index in random.sample(range(len(train_dataset)), 2):
    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

# ============================================================================
# CELL 16: Setup Metrics and Trainer
# ============================================================================

def compute_metrics(p: EvalPrediction):
    predictions = p.predictions
    labels = p.label_ids
    
    # Extract predictions for each task
    hate_type_preds = np.argmax(predictions[0], axis=1)
    hate_severity_preds = np.argmax(predictions[1], axis=1)
    to_whom_preds = np.argmax(predictions[2], axis=1)
    
    # Extract labels for each task
    hate_type_labels = labels[:, 0]
    hate_severity_labels = labels[:, 1]
    to_whom_labels = labels[:, 2]
    
    # Calculate accuracy for each task
    hate_type_acc = accuracy_score(hate_type_labels, hate_type_preds)
    hate_severity_acc = accuracy_score(hate_severity_labels, hate_severity_preds)
    to_whom_acc = accuracy_score(to_whom_labels, to_whom_preds)
    
    # Calculate F1 scores
    hate_type_f1 = f1_score(hate_type_labels, hate_type_preds, average='macro', zero_division=0)
    hate_severity_f1 = f1_score(hate_severity_labels, hate_severity_preds, average='macro', zero_division=0)
    to_whom_f1 = f1_score(to_whom_labels, to_whom_preds, average='macro', zero_division=0)
    
    # Overall accuracy (all three tasks correct)
    overall_acc = np.mean((hate_type_preds == hate_type_labels) & 
                         (hate_severity_preds == hate_severity_labels) & 
                         (to_whom_preds == to_whom_labels))
    
    return {
        "accuracy": overall_acc,
        "hate_type_accuracy": hate_type_acc,
        "hate_severity_accuracy": hate_severity_acc,
        "to_whom_accuracy": to_whom_acc,
        "hate_type_f1": hate_type_f1,
        "hate_severity_f1": hate_severity_f1,
        "to_whom_f1": to_whom_f1,
        "average_f1": (hate_type_f1 + hate_severity_f1 + to_whom_f1) / 3
    }

# Custom data collator for multi-label
class MultiLabelDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        batch = {}
        batch['input_ids'] = torch.tensor([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in features])
        
        if 'hate_type_labels' in features[0]:
            batch['hate_type_labels'] = torch.tensor([f['hate_type_labels'] for f in features])
            batch['hate_severity_labels'] = torch.tensor([f['hate_severity_labels'] for f in features])
            batch['to_whom_labels'] = torch.tensor([f['to_whom_labels'] for f in features])
        
        return batch

data_collator = MultiLabelDataCollator(tokenizer)

if "id" in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns("id")
if "id" in eval_dataset.column_names:
    eval_dataset = eval_dataset.remove_columns("id")

# Custom Trainer for multi-label
class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs['loss']
            
            hate_type_logits = outputs['hate_type_logits']
            hate_severity_logits = outputs['hate_severity_logits']
            to_whom_logits = outputs['to_whom_logits']
            
            predictions = (hate_type_logits, hate_severity_logits, to_whom_logits)
            
            if 'hate_type_labels' in inputs:
                labels = torch.stack([
                    inputs['hate_type_labels'],
                    inputs['hate_severity_labels'],
                    inputs['to_whom_labels']
                ], dim=1)
            else:
                labels = None
        
        return (loss, predictions, labels)

trainer = MultiLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# ============================================================================
# CELL 17: Train Model
# ============================================================================

print("🚀 Starting training with BanglaBERT Large for Multi-Label Classification...")
print(f"📊 Training for {training_args.num_train_epochs} epochs")
print(f"📊 Batch size: {training_args.per_device_train_batch_size}")
print(f"📊 Learning rate: {training_args.learning_rate}")
print(f"📊 Tasks: hate_type, hate_severity, to_whom")

train_result = trainer.train()
metrics = train_result.metrics
max_train_samples = (
    max_train_samples if max_train_samples is not None else len(train_dataset)
)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))

# ============================================================================
# CELL 18: Save Model
# ============================================================================

trainer.save_model()
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

print("✅ Training completed successfully!")
print(f"📊 Training metrics: {metrics}")

## Evaluation and official scorer steps removed per requirement

# ============================================================================
# CELL 20: Generate Predictions
# ============================================================================

logger.info("*** Predict ***")
ids = predict_dataset['id']
predict_dataset = predict_dataset.remove_columns("id")

predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions

# Extract predictions for each task
hate_type_predictions = np.argmax(predictions[0], axis=1)
hate_severity_predictions = np.argmax(predictions[1], axis=1)
to_whom_predictions = np.argmax(predictions[2], axis=1)

# Create output file
output_predict_file = os.path.join(training_args.output_dir, f"subtask_1C_banglabert_large.tsv")

if trainer.is_world_process_zero():
    with open(output_predict_file, "w", encoding='utf-8') as writer:
        logger.info(f"***** Predict results *****")
        writer.write("id\thate_type\thate_severity\tto_whom\tmodel\n")
        for index in range(len(predictions[0])):
            hate_type_label = hate_type_id2l[hate_type_predictions[index]]
            hate_severity_label = hate_severity_id2l[hate_severity_predictions[index]]
            to_whom_label = to_whom_id2l[to_whom_predictions[index]]
            writer.write(f"{ids[index]}\t{hate_type_label}\t{hate_severity_label}\t{to_whom_label}\t{model_name}\n")

print(f"✅ Predictions saved to: {output_predict_file}")

## Omit verbose prediction distribution printing

# ============================================================================
# CELL 21: Show Results
# ============================================================================

print("🎉 Training complete and predictions generated!")
print(f"🤖 Model: {model_name}")
print(f"📁 Output file: {output_predict_file}")
print(f"📁 Model saved in: {training_args.output_dir}")
