# Hate Speech Identification Shared Task: Subtask 1C at BLP Workshop @IJCNLP-AACL 2025
# Multi-label Classification: hate_type, hate_severity, and to_whom
# Modified version using BanglaBERT Large

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================

!pip install transformers datasets evaluate accelerate scikit-learn

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
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
    learning_rate=5e-6,  # Lower learning rate for large model
    num_train_epochs=3,  # Optimal for 35K dataset (prevents overfitting)
    per_device_train_batch_size=4,  # Smaller batch size for large model (335M params)
    per_device_eval_batch_size=4,
    output_dir="./banglabert_large_1C/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    local_rank=-1,
    load_best_model_at_end=True,
    save_total_limit=3,
    save_strategy="epoch",  # Save every epoch
    eval_strategy="epoch",  # Evaluate every epoch
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    warmup_steps=1000,  # More warmup steps for large model
    weight_decay=0.01,  # Add weight decay
    logging_steps=100,
    report_to=None
)

max_train_samples = None
max_eval_samples = None
max_predict_samples = None
max_seq_length = 128  # Reduced for large model memory efficiency
batch_size = 4

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

model_name = 'csebuetnlp/banglabert_large'  # BanglaBERT Large model (335M parameters)
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
    'None': 0,           # For NaN/null values
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
sentence1_key = non_label_column_names[1]  # text column

padding = "max_length"

if max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.")
max_seq_length = min(max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    args = (examples[sentence1_key],)
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    
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
    hate_type_f1 = f1_score(hate_type_labels, hate_type_preds, average='macro')
    hate_severity_f1 = f1_score(hate_severity_labels, hate_severity_preds, average='macro')
    to_whom_f1 = f1_score(to_whom_labels, to_whom_preds, average='macro')
    
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

train_dataset = train_dataset.remove_columns("id")
eval_dataset = eval_dataset.remove_columns("id")

# Custom Trainer for multi-label
class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
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
    tokenizer=tokenizer,
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

# ============================================================================
# CELL 19: Evaluate Model (with Official Scorer)
# ============================================================================

logger.info("*** Evaluate ***")
metrics = trainer.evaluate(eval_dataset=eval_dataset)
max_eval_samples = (
    max_eval_samples if max_eval_samples is not None else len(eval_dataset)
)
metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

print("✅ Evaluation completed!")
print(f"📊 Overall Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
print(f"📊 Hate Type Accuracy: {metrics.get('eval_hate_type_accuracy', 'N/A'):.4f}")
print(f"📊 Hate Severity Accuracy: {metrics.get('eval_hate_severity_accuracy', 'N/A'):.4f}")
print(f"📊 To Whom Accuracy: {metrics.get('eval_to_whom_accuracy', 'N/A'):.4f}")
print(f"📊 Average F1: {metrics.get('eval_average_f1', 'N/A'):.4f}")

# ============================================================================
# OFFICIAL SCORER EVALUATION (using EXACT official scorer from scorer/task.py)
# ============================================================================

# Import official scorer functions directly
import sys
import os
sys.path.append('/content')  # Add current directory to path for Colab

# Copy official scorer functions EXACTLY from scorer/task.py
def _extract_matching_lists_1C(pred_labels, gold_labels):
    """EXACT copy from scorer/task.py"""
    pred_values, gold_values = ({"hate_type": [], "hate_severity": [], "to_whom": []},
                                {"hate_type": [], "hate_severity": [], "to_whom": []})

    for k in gold_labels.keys():
        pred_values["hate_type"].append(pred_labels[k][0])
        pred_values["hate_severity"].append(pred_labels[k][1])
        pred_values["to_whom"].append(pred_labels[k][2])
        gold_values["hate_type"].append(gold_labels[k][0])
        gold_values["hate_severity"].append(gold_labels[k][1])
        gold_values["to_whom"].append(gold_labels[k][2])

    return pred_values, gold_values

def evaluate_1C(pred_labels, gold_labels):
    """EXACT copy from scorer/task.py"""
    pred_values, gold_values = _extract_matching_lists_1C(pred_labels, gold_labels)

    h_acc = accuracy_score(gold_values["hate_type"], pred_values["hate_type"])
    h_precision = precision_score(gold_values["hate_type"], pred_values["hate_type"], average='weighted')
    h_recall = recall_score(gold_values["hate_type"], pred_values["hate_type"], average='weighted')
    h_f1 = f1_score(gold_values["hate_type"], pred_values["hate_type"], average='micro')

    s_acc = accuracy_score(gold_values["hate_severity"], pred_values["hate_severity"])
    s_precision = precision_score(gold_values["hate_severity"], pred_values["hate_severity"], average='weighted')
    s_recall = recall_score(gold_values["hate_severity"], pred_values["hate_severity"], average='weighted')
    s_f1 = f1_score(gold_values["hate_severity"], pred_values["hate_severity"], average='micro')

    w_acc = accuracy_score(gold_values["to_whom"], pred_values["to_whom"])
    w_precision = precision_score(gold_values["to_whom"], pred_values["to_whom"], average='weighted')
    w_recall = recall_score(gold_values["to_whom"], pred_values["to_whom"], average='weighted')
    w_f1 = f1_score(gold_values["to_whom"], pred_values["to_whom"], average='micro')

    acc = (h_acc + s_acc + w_acc) / 3
    precision = (h_precision + s_precision + w_precision) / 3
    recall = (h_recall + s_recall + w_recall) / 3
    f1 = (h_f1 + s_f1 + w_f1) / 3

    return acc, precision, recall, f1

# Run official evaluation on validation data
print("\n🏆 OFFICIAL SCORER EVALUATION (EXACT from scorer/task.py):")
print("=" * 50)

# Prepare data in official scorer format
val_predictions = {}
val_gold_labels = {}

# Get validation predictions
val_pred_results = trainer.predict(eval_dataset, metric_key_prefix="eval")
val_hate_type_preds = np.argmax(val_pred_results.predictions[0], axis=1)
val_hate_severity_preds = np.argmax(val_pred_results.predictions[1], axis=1)
val_to_whom_preds = np.argmax(val_pred_results.predictions[2], axis=1)

# Get validation IDs and labels
val_df_original = pd.read_csv(validation_file, sep='\t')
for idx, row in val_df_original.iterrows():
    doc_id = str(row['id'])
    val_predictions[doc_id] = [
        hate_type_id2l[val_hate_type_preds[idx]],
        hate_severity_id2l[val_hate_severity_preds[idx]], 
        to_whom_id2l[val_to_whom_preds[idx]]
    ]
    val_gold_labels[doc_id] = [
        str(row['hate_type']) if pd.notna(row['hate_type']) else 'None',
        str(row['hate_severity']),
        str(row['to_whom']) if pd.notna(row['to_whom']) else 'None'
    ]

# Run EXACT official evaluation (same as competition)
acc, precision, recall, f1 = evaluate_1C(val_predictions, val_gold_labels)

print(f"📊 OFFICIAL COMPETITION SCORES:")
print(f"   Accuracy: {acc:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1: {f1:.4f}")

print("\n🎯 These are the EXACT metrics the competition will use!")
print("✅ Using identical functions from scorer/task.py")

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

# Show prediction distribution
print("\n📊 Prediction distribution:")
print("Hate Type:")
hate_type_pred_counts = pd.Series(hate_type_predictions).value_counts().sort_index()
for label_id, count in hate_type_pred_counts.items():
    print(f"  {hate_type_id2l[label_id]}: {count}")

print("\nHate Severity:")
hate_severity_pred_counts = pd.Series(hate_severity_predictions).value_counts().sort_index()
for label_id, count in hate_severity_pred_counts.items():
    print(f"  {hate_severity_id2l[label_id]}: {count}")

print("\nTo Whom:")
to_whom_pred_counts = pd.Series(to_whom_predictions).value_counts().sort_index()
for label_id, count in to_whom_pred_counts.items():
    print(f"  {to_whom_id2l[label_id]}: {count}")

# ============================================================================
# CELL 21: Show Results
# ============================================================================

# Display final results
print("🎉 Multi-Label Training and Evaluation Summary")
print("=" * 60)
print(f"🤖 Model: {model_name}")
print(f"📊 Training samples: {len(train_dataset)}")
print(f"📊 Validation samples: {len(eval_dataset)}")
print(f"📊 Test samples: {len(predictions[0])}")
print(f"📊 Overall Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
print(f"📊 Hate Type Accuracy: {metrics.get('eval_hate_type_accuracy', 'N/A'):.4f}")
print(f"📊 Hate Severity Accuracy: {metrics.get('eval_hate_severity_accuracy', 'N/A'):.4f}")
print(f"📊 To Whom Accuracy: {metrics.get('eval_to_whom_accuracy', 'N/A'):.4f}")
print(f"📊 Average F1 Score: {metrics.get('eval_average_f1', 'N/A'):.4f}")
print(f"📁 Output file: {output_predict_file}")
print(f"📁 Model saved in: {training_args.output_dir}")

print("\n✅ All done! Multi-label model trained and predictions generated!")
print("This model simultaneously predicts hate_type, hate_severity, and to_whom!")
