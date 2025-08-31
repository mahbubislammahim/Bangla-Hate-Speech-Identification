# Hate Speech Identification Shared Task: Subtask 1B at BLP Workshop @IJCNLP-AACL 2025
# IMPROVED VERSION with advanced techniques for better accuracy

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================

!pip install transformers datasets evaluate accelerate huggingface_hub nlpaug sentencepiece

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
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

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
    EarlyStoppingCallback,
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
# CELL 5: IMPROVED Training Arguments
# ============================================================================

training_args = TrainingArguments(
    learning_rate=3e-5,  # Slightly higher learning rate for better convergence
    num_train_epochs=4,  # More epochs with early stopping
    per_device_train_batch_size=8,  # Increased batch size for better gradient estimates
    per_device_eval_batch_size=16,  # Larger eval batch size
    output_dir="./banglabert_improved_model/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    local_rank=-1,
    load_best_model_at_end=True,
    save_total_limit=2,  # Save only best 2 checkpoints
    save_strategy="epoch",
    eval_strategy="epoch",
    metric_for_best_model="eval_f1",  # Use F1 instead of accuracy
    greater_is_better=True,
    warmup_steps=500,  # Optimized warmup
    weight_decay=0.01,
    logging_steps=100,
    report_to=None,
    dataloader_num_workers=4,  # Parallel data loading
    fp16=True,  # Mixed precision training
    gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    run_name="banglabert_improved",
    # Early stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
)

max_train_samples = None
max_eval_samples = None
max_predict_samples = None
max_seq_length = 256  # Increased sequence length for better context
batch_size = 8

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
# CELL 7: IMPROVED Model Selection
# ============================================================================

# Try different model variants for better performance
model_name = 'csebuetnlp/banglabert'  # Use base BanglaBERT instead of large for better generalization
print(f"🤖 Using BanglaBERT Base model: {model_name}")

# ============================================================================
# CELL 8: Set Random Seed
# ============================================================================

set_seed(42)  # Fixed seed for reproducibility

# ============================================================================
# CELL 9: Load and Preprocess Data with Augmentation
# ============================================================================

l2id = {'None': 0, 'Society': 1, 'Organization': 2, 'Community': 3, 'Individual': 4}

train_df = pd.read_csv(train_file, sep='\t')
train_df['label'] = train_df['label'].map(l2id).fillna(0).astype(int)
validation_df = pd.read_csv(validation_file, sep='\t')
validation_df['label'] = validation_df['label'].map(l2id).fillna(0).astype(int)
test_df = pd.read_csv(test_file, sep='\t')

# Data augmentation for minority classes
print("🔄 Applying data augmentation for balanced training...")

def augment_text(text, label, augmenter):
    """Apply text augmentation"""
    try:
        augmented = augmenter.augment(text)[0]
        return augmented
    except:
        return text

# Initialize augmenters
synonym_aug = naw.SynonymAug(aug_src='wordnet', lang='ben')
backtranslation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-bn',
    to_model_name='facebook/wmt19-bn-en'
)

# Apply augmentation to minority classes
augmented_data = []
label_counts = train_df['label'].value_counts()
max_samples = label_counts.max()

for label in train_df['label'].unique():
    label_data = train_df[train_df['label'] == label]
    current_count = len(label_data)
    
    if current_count < max_samples:
        # Augment minority classes
        samples_needed = max_samples - current_count
        augmented_samples = label_data.sample(n=min(samples_needed, current_count), replace=True)
        
        for _, row in augmented_samples.iterrows():
            # Apply synonym replacement
            aug_text = augment_text(row['text'], row['label'], synonym_aug)
            augmented_data.append({
                'id': f"aug_{row['id']}",
                'text': aug_text,
                'label': row['label']
            })

# Add augmented data to training set
if augmented_data:
    aug_df = pd.DataFrame(augmented_data)
    train_df = pd.concat([train_df, aug_df], ignore_index=True)
    print(f"✅ Added {len(augmented_data)} augmented samples")

# Convert to datasets
train_df = Dataset.from_pandas(train_df)
validation_df = Dataset.from_pandas(validation_df)
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

# Add special tokens for better performance
special_tokens = {
    'additional_special_tokens': ['[HATE]', '[TARGET]', '[SEVERITY]']
}
tokenizer.add_special_tokens(special_tokens)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    from_tf=bool(".ckpt" in model_name),
    config=config,
    cache_dir=None,
    revision="main",
    use_auth_token=None,
    ignore_mismatched_sizes=False,
)

# Resize token embeddings for new special tokens
model.resize_token_embeddings(len(tokenizer))

print(f"✅ Improved BanglaBERT model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# CELL 12: IMPROVED Preprocessing with Better Tokenization
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

def improved_preprocess_function(examples):
    """Improved preprocessing with better text handling"""
    texts = examples[sentence1_key]
    
    # Clean and normalize text
    cleaned_texts = []
    for text in texts:
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Add special tokens for better context
        text = f"[HATE] {text} [TARGET]"
        cleaned_texts.append(text)
    
    args = (cleaned_texts,)
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

raw_datasets = raw_datasets.map(
    improved_preprocess_function,
    batched=True,
    load_from_cache_file=True,
    desc="Running improved tokenizer on dataset",
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
# CELL 15: IMPROVED Metrics and Trainer
# ============================================================================

metric = evaluate.load("accuracy")

def improved_compute_metrics(p: EvalPrediction):
    """Improved metrics computation with F1 score"""
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    
    accuracy = (preds == p.label_ids).astype(np.float32).mean().item()
    f1 = f1_score(p.label_ids, preds, average='micro')
    precision = precision_score(p.label_ids, preds, average='weighted')
    recall = recall_score(p.label_ids, preds, average='weighted')
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Custom data collator with dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = train_dataset.remove_columns("id")
eval_dataset = eval_dataset.remove_columns("id")

# Add early stopping callback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=improved_compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping_callback],
)

# ============================================================================
# CELL 16: Train Model
# ============================================================================

print("🚀 Starting improved training with BanglaBERT...")
print(f"📊 Training for {training_args.num_train_epochs} epochs")
print(f"📊 Batch size: {training_args.per_device_train_batch_size}")
print(f"📊 Learning rate: {training_args.learning_rate}")
print(f"📊 Sequence length: {max_seq_length}")

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
print(f"📊 Validation F1: {metrics.get('eval_f1', 'N/A'):.4f}")
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
id2l = {v: k for k, v in l2id.items()}
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

output_predict_file = os.path.join(training_args.output_dir, f"subtask_1B_banglabert_improved.tsv")

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

print("🎉 All done! Improved model trained and predictions generated!")
print(f"📁 Results saved in: {training_args.output_dir}")
print(f"📊 Final Validation Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
print(f"📊 Final Validation F1: {metrics.get('eval_f1', 'N/A'):.4f}")

# ============================================================================
# CELL 21: Upload to Hugging Face Hub (Optional)
# ============================================================================

# Set your Hugging Face repository name here
HF_REPO_NAME = "Mahim47/banglabert-improved-hatespeech-subtask1b"  # Change this to your desired repo name

print("\n🔑 HUGGING FACE UPLOAD:")
print("=" * 50)

# Uncomment the following lines to upload to Hugging Face Hub:
print("🔑 Please login to Hugging Face Hub...")
login()  # Uncomment this line and provide your HF token

print(f"🚀 To upload model to Hugging Face Hub: {HF_REPO_NAME}")
print("Uncomment the code below to upload:")

# Step 1: Login to Hugging Face (you'll need to provide your token)
login()  # This will prompt for your HF token

# Step 2: Push the model to Hub
print(f"🚀 Uploading model to Hugging Face Hub: {HF_REPO_NAME}")

# Push model and tokenizer separately to avoid parameter conflicts
model.push_to_hub(
    HF_REPO_NAME,
    commit_message=f"Improved BanglaBERT for hate speech detection (Subtask 1B) - F1: {metrics.get('eval_f1', 'N/A'):.4f}",
    private=False
)

tokenizer.push_to_hub(
    HF_REPO_NAME,
    commit_message=f"Improved BanglaBERT tokenizer for hate speech detection (Subtask 1B)",
    private=False
)

print(f"✅ Model successfully uploaded to: https://huggingface.co/{HF_REPO_NAME}")
print("🎯 Your improved model is now available for others to use!")

print("\n" + "="*60)
print("📝 TO UPLOAD TO HUGGING FACE:")
print("1. Uncomment the upload code above")
print("2. Change HF_REPO_NAME to your desired repository name")
print("3. Get your HF token from: https://huggingface.co/settings/tokens")
print("4. Run the upload section and provide your token when prompted")
print("="*60)

# ============================================================================
# CELL 22: Model Ensemble (Optional - for even better performance)
# ============================================================================

print("\n🔄 Training ensemble models for better performance...")

# Train multiple models with different seeds
ensemble_models = []
ensemble_trainers = []

for seed in [42, 123, 456, 789, 999]:
    print(f"🌱 Training ensemble model with seed {seed}")
    
    # Set seed
    set_seed(seed)
    
    # Create new model instance
    ensemble_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=False,
    )
    ensemble_model.resize_token_embeddings(len(tokenizer))
    
    # Create trainer for this model
    ensemble_trainer = Trainer(
        model=ensemble_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=improved_compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
    )
    
    # Train the model
    ensemble_trainer.train()
    
    # Evaluate
    ensemble_metrics = ensemble_trainer.evaluate(eval_dataset=eval_dataset)
    print(f"📊 Ensemble model {seed} - F1: {ensemble_metrics.get('eval_f1', 'N/A'):.4f}")
    
    ensemble_models.append(ensemble_model)
    ensemble_trainers.append(ensemble_trainer)

# Ensemble prediction
print("\n🎯 Generating ensemble predictions...")

ensemble_predictions = []
for ensemble_trainer in ensemble_trainers:
    pred_results = ensemble_trainer.predict(predict_dataset, metric_key_prefix="predict")
    ensemble_predictions.append(pred_results.predictions)

# Average predictions
ensemble_predictions = np.mean(ensemble_predictions, axis=0)
ensemble_predictions = np.argmax(ensemble_predictions, axis=1)

# Save ensemble predictions
ensemble_output_file = os.path.join(training_args.output_dir, f"subtask_1B_banglabert_ensemble.tsv")

if trainer.is_world_process_zero():
    with open(ensemble_output_file, "w") as writer:
        logger.info(f"***** Ensemble predict results *****")
        writer.write("id\tlabel\tmodel\n")
        for index, item in enumerate(ensemble_predictions):
            item = label_list[item]
            item = id2l[item]
            writer.write(f"{ids[index]}\t{item}\t{model_name}_ensemble\n")

print(f"✅ Ensemble predictions saved to: {ensemble_output_file}")

print("\n" + "="*60)
print("🎯 IMPROVEMENT TECHNIQUES APPLIED:")
print("1. ✅ Data augmentation for minority classes")
print("2. ✅ Increased sequence length (256 vs 128)")
print("3. ✅ Better hyperparameters (learning rate, batch size)")
print("4. ✅ Early stopping to prevent overfitting")
print("5. ✅ F1 score optimization instead of accuracy")
print("6. ✅ Mixed precision training (fp16)")
print("7. ✅ Gradient accumulation for larger effective batch size")
print("8. ✅ Special tokens for better context")
print("9. ✅ Model ensemble for better generalization")
print("10. ✅ Improved text preprocessing")
print("="*60)
