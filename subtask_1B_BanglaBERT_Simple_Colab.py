# Hate Speech Identification Shared Task: Subtask 1B at BLP Workshop @IJCNLP-AACL 2025
# Modified version using BanglaBERT instead of DistilBERT

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================

!pip install transformers datasets evaluate accelerate huggingface_hub

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
from huggingface_hub import HfApi, login

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

train_file = 'blp25_hatespeech_subtask_1B_train.tsv'
validation_file = 'blp25_hatespeech_subtask_1B_dev.tsv'
test_file = 'blp25_hatespeech_subtask_1B_dev_test.tsv'

# ============================================================================
# CELL 5: Training Arguments (Optimized for BanglaBERT)
# ============================================================================

training_args = TrainingArguments(
    learning_rate=5e-6,  # Lower learning rate for large model
    num_train_epochs=3,  # Optimal for 35K dataset (prevents overfitting)
    per_device_train_batch_size=4,  # Smaller batch size for large model (335M params)
    per_device_eval_batch_size=4,
    output_dir="./banglabert_large_model/",
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
# CELL 7: Define Model (CHANGED TO BANGLABERT LARGE)
# ============================================================================

model_name = 'csebuetnlp/banglabert_large'  # BanglaBERT Large model (335M parameters)
print(f"🤖 Using BanglaBERT Large model: {model_name}")

# ============================================================================
# CELL 8: Set Random Seed
# ============================================================================

set_seed(training_args.seed)

# ============================================================================
# CELL 9: Load Data Files
# ============================================================================

l2id = {'None': 0, 'Society': 1, 'Organization': 2, 'Community': 3, 'Individual': 4}

train_df = pd.read_csv(train_file, sep='\t')
train_df['label'] = train_df['label'].map(l2id).fillna(0).astype(int)
train_df = Dataset.from_pandas(train_df)

validation_df = pd.read_csv(validation_file, sep='\t')
validation_df['label'] = validation_df['label'].map(l2id).fillna(0).astype(int)
validation_df = Dataset.from_pandas(validation_df)

test_df = pd.read_csv(test_file, sep='\t')
test_df = Dataset.from_pandas(test_df)

data_files = {"train": train_df, "validation": validation_df, "test": test_df}
for key in data_files.keys():
    logger.info(f"loading a local file for {key}")

raw_datasets = DatasetDict(
    {"train": train_df, "validation": validation_df, "test": test_df}
)

# ============================================================================
# CELL 10: Extract Labels
# ============================================================================

label_list = raw_datasets["train"].unique("label")
print(f"Label list: {label_list}")
label_list.sort()
num_labels = len(label_list)
logger.info(f"Number of labels: {num_labels}")

# ============================================================================
# CELL 11: Load Model, Tokenizer, and Config
# ============================================================================

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=num_labels,
    finetuning_task=None,
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

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    from_tf=bool(".ckpt" in model_name),
    config=config,
    cache_dir=None,
    revision="main",
    use_auth_token=None,
    ignore_mismatched_sizes=False,
)

print(f"✅ Large BanglaBERT model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# CELL 12: Preprocess Data
# ============================================================================

non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
sentence1_key = non_label_column_names[1]

padding = "max_length"

label_to_id = None
if (model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id):
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if sorted(label_name_to_id.keys()) == sorted(label_list):
        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
        logger.warning(
            "Your model seems to have been trained with labels, but they don't match the dataset: ",
            f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
            "\nIgnoring the model labels as a result.",)

if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

if max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.")
max_seq_length = min(max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    args = (examples[sentence1_key],)
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

# ============================================================================
# CELL 13: Prepare Datasets
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

if "test" not in raw_datasets and "test_matched" not in raw_datasets:
    raise ValueError("requires a test dataset")
predict_dataset = raw_datasets["test"]
if max_predict_samples is not None:
    max_predict_samples_n = min(len(predict_dataset), max_predict_samples)
    predict_dataset = predict_dataset.select(range(max_predict_samples_n))

# ============================================================================
# CELL 14: Log Sample Data
# ============================================================================

for index in random.sample(range(len(train_dataset)), 3):
    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

# ============================================================================
# CELL 15: Setup Metrics and Trainer
# ============================================================================

metric = evaluate.load("accuracy")

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

data_collator = default_data_collator

train_dataset = train_dataset.remove_columns("id")
eval_dataset = eval_dataset.remove_columns("id")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ============================================================================
# CELL 16: Train Model
# ============================================================================

print("🚀 Starting training with BanglaBERT...")
print(f"📊 Training for {training_args.num_train_epochs} epochs")
print(f"📊 Batch size: {training_args.per_device_train_batch_size}")
print(f"📊 Learning rate: {training_args.learning_rate}")

train_result = trainer.train()
metrics = train_result.metrics
max_train_samples = (
    max_train_samples if max_train_samples is not None else len(train_dataset)
)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))

# ============================================================================
# CELL 17: Save Model
# ============================================================================

trainer.save_model()
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

print("✅ Training completed successfully!")
print(f"📊 Training metrics: {metrics}")

# ============================================================================
# CELL 18: Evaluate Model
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
print(f"📊 Validation Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
print(f"📊 Validation Loss: {metrics.get('eval_loss', 'N/A'):.4f}")

# ============================================================================
# OFFICIAL SCORER EVALUATION (EXACT from scorer/task.py)
# ============================================================================

# Official scorer functions EXACTLY copied from scorer/task.py
def _extract_matching_lists(pred_labels, gold_labels, subtask):
    """EXACT copy from scorer/task.py"""
    pred_values, gold_values = ([], [])
    for k in gold_labels.keys():
        pred_values.append(pred_labels[k])
        gold_values.append(gold_labels[k])
    return pred_values, gold_values

def evaluate_official(pred_labels, gold_labels, subtask):
    """EXACT copy from scorer/task.py"""
    pred_values, gold_values = _extract_matching_lists(pred_labels, gold_labels, subtask)
    acc = accuracy_score(gold_values, pred_values)
    precision = precision_score(gold_values, pred_values, average='weighted')
    recall = recall_score(gold_values, pred_values, average='weighted')
    f1 = f1_score(gold_values, pred_values, average='micro')
    return acc, precision, recall, f1

# Run official evaluation on validation data
print("\n🏆 OFFICIAL SCORER EVALUATION (EXACT from scorer/task.py):")
print("=" * 50)

# Get validation predictions
val_pred_results = trainer.predict(eval_dataset, metric_key_prefix="eval")
val_predictions_raw = np.argmax(val_pred_results.predictions, axis=1)

# Prepare data in official scorer format
val_predictions = {}
val_gold_labels = {}

# Get validation IDs and labels
val_df_original = pd.read_csv(validation_file, sep='\t')
for idx, row in val_df_original.iterrows():
    doc_id = str(row['id'])
    val_predictions[doc_id] = id2l[val_predictions_raw[idx]]
    val_gold_labels[doc_id] = str(row['label'])

# Run EXACT official evaluation (same as competition)
acc, precision, recall, f1 = evaluate_official(val_predictions, val_gold_labels, '1B')

print(f"📊 OFFICIAL COMPETITION SCORES:")
print(f"   Accuracy: {acc:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1: {f1:.4f}")

print("\n🎯 These are the EXACT metrics the competition will use!")
print("✅ Using identical functions from scorer/task.py")

# ============================================================================
# CELL 19: Generate Predictions
# ============================================================================

id2l = {v: k for k, v in l2id.items()}
logger.info("*** Predict ***")
ids = predict_dataset['id']
predict_dataset = predict_dataset.remove_columns("id")
predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
predictions = np.argmax(predictions, axis=1)

output_predict_file = os.path.join(training_args.output_dir, f"subtask_1B_banglabert_large.tsv")

if trainer.is_world_process_zero():
    with open(output_predict_file, "w") as writer:
        logger.info(f"***** Predict results *****")
        writer.write("id\tlabel\tmodel\n")
        for index, item in enumerate(predictions):
            item = label_list[item]
            item = id2l[item]
            writer.write(f"{ids[index]}\t{item}\t{model_name}\n")

print(f"✅ Predictions saved to: {output_predict_file}")

# ============================================================================
# CELL 20: Save Model Card
# ============================================================================

kwargs = {"finetuned_from": model_name, "tasks": "text-classification"}
trainer.create_model_card(**kwargs)

print("🎉 All done! Model trained and predictions generated!")
print(f"📁 Results saved in: {training_args.output_dir}")
print(f"📊 Final Validation Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")

# ============================================================================
# CELL 21: Upload to Hugging Face Hub (Optional)
# ============================================================================

# Set your Hugging Face repository name here
HF_REPO_NAME = "Mahim47/banglabert-hatespeech-subtask1b"  # Change this to your desired repo name

# Uncomment the following lines to upload to Hugging Face Hub:
# 
# # Step 1: Login to Hugging Face (you'll need to provide your token)
print("🔑 Please login to Hugging Face Hub...")
login()  # This will prompt for your HF token
# 
# # Step 2: Push the model to Hub
print(f"🚀 Uploading model to Hugging Face Hub: {HF_REPO_NAME}")

# Push model and tokenizer separately to avoid parameter conflicts
model.push_to_hub(
    HF_REPO_NAME,
    commit_message=f"Fine-tuned BanglaBERT for hate speech detection (Subtask 1B) - Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}",
    private=False
)

tokenizer.push_to_hub(
    HF_REPO_NAME,
    commit_message=f"Fine-tuned BanglaBERT tokenizer for hate speech detection (Subtask 1B)",
    private=False
)

print(f"✅ Model successfully uploaded to: https://huggingface.co/{HF_REPO_NAME}")
print("🎯 Your model is now available for others to use!")

print("\n" + "="*60)
print("📝 TO UPLOAD TO HUGGING FACE:")
print("1. Uncomment the upload code above")
print("2. Change HF_REPO_NAME to your desired repository name")
print("3. Get your HF token from: https://huggingface.co/settings/tokens")
print("4. Run the cell and provide your token when prompted")
print("="*60)

# ============================================================================
# CELL 22: Continue Training for Additional Epochs (Optional)
# ============================================================================

print("🔄 Continuing training from checkpoint for 1 more epoch...")

# Specify your checkpoint path
CHECKPOINT_PATH = "./banglabert_large_model/checkpoint-17762"

# Update training arguments for continued training
continued_training_args = TrainingArguments(
    learning_rate=2e-6,  # Lower learning rate for continued training
    num_train_epochs=3,  # Total epochs (will continue from where checkpoint left off)
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir="./banglabert_large_model_continued/",  # New output directory
    overwrite_output_dir=True,
    remove_unused_columns=False,
    local_rank=-1,
    load_best_model_at_end=True,
    save_total_limit=3,
    save_strategy="epoch",
    eval_strategy="epoch",
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    warmup_steps=100,  # Fewer warmup steps for continued training
    weight_decay=0.01,
    logging_steps=50,
    report_to=None
)

# Create new trainer with continued training arguments
continued_trainer = Trainer(
    model=model,
    args=continued_training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Load and evaluate current checkpoint performance
print(f"📂 Loading checkpoint: {CHECKPOINT_PATH}")

# Continue training from specific checkpoint
continued_train_result = continued_trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
continued_metrics = continued_train_result.metrics

# Save the continued model
continued_trainer.save_model()
continued_trainer.log_metrics("continued_train", continued_metrics)
continued_trainer.save_metrics("continued_train", continued_metrics)
continued_trainer.save_state()

# Evaluate the continued model
print("🔍 Evaluating continued model...")
continued_eval_metrics = continued_trainer.evaluate(eval_dataset=eval_dataset)
continued_trainer.log_metrics("continued_eval", continued_eval_metrics)
continued_trainer.save_metrics("continued_eval", continued_eval_metrics)

print("✅ Continued training completed!")
print(f"📊 Original Validation Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
print(f"📊 New Validation Accuracy: {continued_eval_metrics.get('eval_accuracy', 'N/A'):.4f}")
print(f"📈 Improvement: {continued_eval_metrics.get('eval_accuracy', 0) - metrics.get('eval_accuracy', 0):.4f}")

# ============================================================================
# CELL 23: Generate Predictions with Continued Model
# ============================================================================

print("\n🔮 Generating predictions with continued model...")

# Recreate predict_dataset for continued training (since it was modified earlier)
continued_predict_dataset_original = raw_datasets["test"]
if max_predict_samples is not None:
    max_predict_samples_n = min(len(continued_predict_dataset_original), max_predict_samples)
    continued_predict_dataset_original = continued_predict_dataset_original.select(range(max_predict_samples_n))

# Use the continued trainer for predictions
continued_ids = continued_predict_dataset_original['id']
continued_predict_dataset = continued_predict_dataset_original.remove_columns("id")
continued_predictions = continued_trainer.predict(continued_predict_dataset, metric_key_prefix="predict").predictions
continued_predictions = np.argmax(continued_predictions, axis=1)

# Save predictions from continued model
continued_output_file = os.path.join(continued_training_args.output_dir, f"subtask_1B_banglabert_large_continued.tsv")

if continued_trainer.is_world_process_zero():
    with open(continued_output_file, "w") as writer:
        logger.info(f"***** Continued model predict results *****")
        writer.write("id\tlabel\tmodel\n")
        for index, item in enumerate(continued_predictions):
            item = label_list[item]
            item = id2l[item]
            writer.write(f"{continued_ids[index]}\t{item}\t{model_name}\n")

print(f"✅ Continued model predictions saved to: {continued_output_file}")

# ============================================================================
# CELL 24: Official Scorer Evaluation for Continued Model
# ============================================================================

print("\n🏆 OFFICIAL SCORER EVALUATION for Continued Model:")
print("=" * 50)

# Get validation predictions from continued model
continued_val_pred_results = continued_trainer.predict(eval_dataset, metric_key_prefix="eval")
continued_val_predictions_raw = np.argmax(continued_val_pred_results.predictions, axis=1)

# Prepare data in official scorer format
continued_val_predictions = {}
continued_val_gold_labels = {}

# Get validation IDs and labels
continued_val_df_original = pd.read_csv(validation_file, sep='\t')
for idx, row in continued_val_df_original.iterrows():
    doc_id = str(row['id'])
    continued_val_predictions[doc_id] = id2l[continued_val_predictions_raw[idx]]
    continued_val_gold_labels[doc_id] = str(row['label'])

# Run EXACT official evaluation for continued model
continued_acc, continued_precision, continued_recall, continued_f1 = evaluate_official(continued_val_predictions, continued_val_gold_labels, '1B')

print(f"📊 CONTINUED MODEL OFFICIAL COMPETITION SCORES:")
print(f"   Accuracy: {continued_acc:.4f}")
print(f"   Precision: {continued_precision:.4f}")
print(f"   Recall: {continued_recall:.4f}")
print(f"   F1: {continued_f1:.4f}")

print(f"\n📈 IMPROVEMENT COMPARISON:")
print(f"   Original Accuracy: {acc:.4f}")
print(f"   Continued Accuracy: {continued_acc:.4f}")
print(f"   Accuracy Improvement: {continued_acc - acc:.4f}")

print("\n🎯 These are the EXACT metrics for your continued model!")
print("✅ Using identical functions from scorer/task.py")

# ============================================================================
# CELL 25: Upload Continued Model to Hugging Face Hub (Optional)
# ============================================================================

# Set your Hugging Face repository name for the continued model
HF_REPO_NAME_CONTINUED = "Mahim47/banglabert-hatespeech-subtask1b-v2"  # Version 2

print(f"\n🚀 Uploading continued model to Hugging Face Hub: {HF_REPO_NAME_CONTINUED}")

# Push continued model and tokenizer (using the same model/tokenizer objects)
model.push_to_hub(
    HF_REPO_NAME_CONTINUED,
    commit_message=f"Fine-tuned BanglaBERT v2 for hate speech detection (Subtask 1B) - Accuracy: {continued_eval_metrics.get('eval_accuracy', 'N/A'):.4f}",
    private=False
)

tokenizer.push_to_hub(
    HF_REPO_NAME_CONTINUED,
    commit_message=f"Fine-tuned BanglaBERT v2 tokenizer for hate speech detection (Subtask 1B)",
    private=False
)

print(f"✅ Continued model successfully uploaded to: https://huggingface.co/{HF_REPO_NAME_CONTINUED}")
print("🎯 Your improved model is now available for others to use!")

print("\n" + "="*50)
print("🔄 TO CONTINUE TRAINING:")
print("1. Uncomment the continued training code above")
print("2. Adjust learning rate if needed (currently set to 2e-6)")
print("3. Run the cell to train for 1 more epoch")
print("="*50)
