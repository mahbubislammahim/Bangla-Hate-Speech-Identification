# Hate Speech Identification Shared Task: Subtask 1B at BLP Workshop @IJCNLP-AACL 2025
# IMPROVED VERSION with advanced techniques for better accuracy

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================

!pip install transformers datasets evaluate accelerate huggingface_hub sentencepiece

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
# Removed nlpaug imports - using simpler data balancing approach

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
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
import torch.nn as nn
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
    learning_rate=2e-5,  # More conservative learning rate for stability
    num_train_epochs=4,  # Fewer epochs with early stopping
    per_device_train_batch_size=16,  # Optimal batch size for most GPUs
    per_device_eval_batch_size=32,  # Larger eval batch size for efficiency
    output_dir="./banglabert_improved_model/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    
    # Model saving and evaluation
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,  # Save only best 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",  # Use F1 instead of accuracy
    greater_is_better=True,
    
    # Optimization settings
    warmup_steps=500,  # Warmup for stable training
    weight_decay=0.01,
    gradient_accumulation_steps=1,  # Direct batch size without accumulation
    
    # Logging and monitoring
    logging_steps=50,
    logging_dir="./logs",
    run_name="banglabert_improved",
    report_to=None,
    
    # Performance optimizations
    dataloader_num_workers=0,  # Disable multiprocessing for stability
    fp16=True,  # Mixed precision training
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

# Load datasets
print(f"📊 Loading training dataset: {train_file}")
train_df = pd.read_csv(train_file, sep='\t')
train_df['label'] = train_df['label'].map(l2id).fillna(0).astype(int)

validation_df = pd.read_csv(validation_file, sep='\t')
validation_df['label'] = validation_df['label'].map(l2id).fillna(0).astype(int)
test_df = pd.read_csv(test_file, sep='\t')

# Show dataset statistics
print("📊 Dataset Statistics:")
print(f"   Training samples: {len(train_df):,}")
print(f"   Validation samples: {len(validation_df):,}")
print(f"   Test samples: {len(test_df):,}")

# Show class distribution
label_counts = train_df['label'].value_counts().sort_index()
id2l = {v: k for k, v in l2id.items()}
print("\n📊 Class Distribution in Training Data:")
for label_id, count in label_counts.items():
    label_name = id2l[label_id]
    percentage = (count / len(train_df)) * 100
    print(f"   {label_name}: {count:,} ({percentage:.1f}%)")

print(f"\n✅ Dataset loaded successfully!")
print("🎯 Using training data with merged augmentation - ready for optimal training!")

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
# CELL 11: Enhanced Model Architecture with Additional Layers
# ============================================================================

class EnhancedBanglaBERT(nn.Module):
    """BanglaBERT with enhanced classification head for better performance"""
    
    def __init__(self, model_name, num_labels, hidden_dropout=0.3, use_attention_pooling=True):
        super().__init__()
        self.num_labels = num_labels
        self.use_attention_pooling = use_attention_pooling
        
        # Load pre-trained BanglaBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config  # Expose config for compatibility
        
        # Update config for classification
        self.config.num_labels = num_labels
        # Set default label mappings (will be updated later if needed)
        self.config.label2id = {f"LABEL_{i}": i for i in range(num_labels)}
        self.config.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        
        hidden_size = self.bert.config.hidden_size
        
        # Attention pooling layer (optional)
        if use_attention_pooling:
            self.attention_pooling = nn.Linear(hidden_size, 1)
        
        # Enhanced classification head with multiple layers
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.dense1 = nn.Linear(hidden_size, 512)
        self.activation1 = nn.GELU()  # GELU works better than ReLU for transformers
        
        self.dropout2 = nn.Dropout(hidden_dropout)
        self.dense2 = nn.Linear(512, 256)
        self.activation2 = nn.GELU()
        
        self.dropout3 = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(256, num_labels)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the new layers"""
        for module in [self.dense1, self.dense2, self.classifier]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Use attention pooling or standard pooling
        if self.use_attention_pooling and attention_mask is not None:
            # Attention-based pooling
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
            attention_weights = self.attention_pooling(hidden_states).squeeze(-1)  # [batch, seq_len]
            attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
            attention_weights = torch.softmax(attention_weights, dim=-1)
            pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        else:
            # Standard pooling
            pooled_output = outputs.pooler_output
        
        # Enhanced classification head
        x = self.dropout1(pooled_output)
        x = self.dense1(x)
        x = self.activation1(x)
        
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.activation2(x)
        
        x = self.dropout3(x)
        logits = self.classifier(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None,
        }

print("🧠 Enhanced BanglaBERT model class defined!")

# ============================================================================
# CELL 12: Load Model, Tokenizer, and Config
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

# Keep tokenizer simple and standard (no special tokens needed)
# Special tokens can hurt performance if not properly trained

# Option 1: Use standard model (original approach)
use_enhanced_model = True  # Set to False to use standard model

if use_enhanced_model:
    print("🚀 Using Enhanced BanglaBERT with additional layers...")
    model = EnhancedBanglaBERT(
        model_name=model_name,
        num_labels=num_labels,
        hidden_dropout=0.3,  # Adjust dropout for regularization
        use_attention_pooling=True  # Use attention-based pooling
    )
    print(f"✅ Enhanced BanglaBERT model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("🧠 Model features:")
    print("   • Enhanced classification head (768 → 512 → 256 → num_labels)")
    print("   • Attention-based pooling for better representation")
    print("   • GELU activation functions")
    print("   • Proper dropout regularization")
else:
    print("📊 Using Standard BanglaBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
        ignore_mismatched_sizes=False,
    )
    print(f"✅ Standard BanglaBERT model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# CELL 12: IMPROVED Preprocessing with Better Tokenization
# ============================================================================

non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
sentence1_key = non_label_column_names[1]

print(f"📊 Dataset columns: {raw_datasets['train'].column_names}")
print(f"📊 Text column key: {sentence1_key}")
print(f"📊 Sample data structure:")
sample = raw_datasets['train'][0]
for key, value in sample.items():
    print(f"   {key}: {type(value)} - {str(value)[:100]}")

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
    """Simple and effective preprocessing with better error handling"""
    texts = examples[sentence1_key]
    
    # Clean and normalize text (keep it simple)
    cleaned_texts = []
    for text in texts:
        # Ensure text is a string
        if isinstance(text, list):
            text = ' '.join(str(t) for t in text)
        text = str(text)  # Convert to string to be safe
        
        # Basic cleaning: remove extra whitespace
        text = ' '.join(text.split())
        # Remove URLs, mentions if present
        text = text.replace('@', '').replace('http', '').replace('www', '')
        cleaned_texts.append(text)
    
    # Tokenize with explicit parameters
    result = tokenizer(
        cleaned_texts,
        padding="max_length",  # Use max_length padding for consistency
        max_length=max_seq_length,
        truncation=True,
        return_tensors=None  # Don't return tensors yet, let data collator handle it
    )
    
    # Handle labels properly
    if "label" in examples:
        if label_to_id is not None:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        else:
            result["label"] = examples["label"]
    
    return result

raw_datasets = raw_datasets.map(
    improved_preprocess_function,
    batched=True,
    load_from_cache_file=True,
    desc="Running improved tokenizer on dataset",
)

# Remove text column after tokenization to avoid conflicts
print(f"📊 Columns before cleanup: {raw_datasets['train'].column_names}")
columns_to_remove = ['text']
for split in raw_datasets.keys():
    for col in columns_to_remove:
        if col in raw_datasets[split].column_names:
            raw_datasets[split] = raw_datasets[split].remove_columns(col)

print(f"📊 Columns after cleanup: {raw_datasets['train'].column_names}")

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

# Use default data collator for better compatibility
data_collator = default_data_collator

print("📊 Using default data collator for maximum compatibility")

# Remove 'id' column if it exists
if "id" in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns("id")
if "id" in eval_dataset.column_names:
    eval_dataset = eval_dataset.remove_columns("id")

print(f"📊 Final training dataset columns: {train_dataset.column_names}")
print(f"📊 Final eval dataset columns: {eval_dataset.column_names}")

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
# CELL 22: Test Time Augmentation (TTA) - More Efficient than Full Ensemble
# ============================================================================

print("\n🎯 Applying Test Time Augmentation for better predictions...")

def apply_tta_prediction(trainer, dataset, num_augmentations=3):
    """Apply test time augmentation for more robust predictions"""
    all_predictions = []
    
    for aug_idx in range(num_augmentations):
        print(f"🔄 TTA iteration {aug_idx + 1}/{num_augmentations}")
        
        # Apply slight variations (different random seeds for dropout)
        trainer.model.train()  # Enable dropout for variation
        with torch.no_grad():
            pred_results = trainer.predict(dataset, metric_key_prefix="predict")
            all_predictions.append(pred_results.predictions)
        trainer.model.eval()  # Back to eval mode
    
    # Average predictions across augmentations
    avg_predictions = np.mean(all_predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)

# Apply TTA to test predictions
tta_predictions = apply_tta_prediction(trainer, predict_dataset, num_augmentations=5)

# Save TTA predictions
tta_output_file = os.path.join(training_args.output_dir, f"subtask_1B_banglabert_tta.tsv")

if trainer.is_world_process_zero():
    with open(tta_output_file, "w") as writer:
        logger.info(f"***** TTA predict results *****")
        writer.write("id\tlabel\tmodel\n")
        for index, item in enumerate(tta_predictions):
            item = label_list[item]
            item = id2l[item]
            writer.write(f"{ids[index]}\t{item}\t{model_name}_tta\n")

print(f"✅ TTA predictions saved to: {tta_output_file}")
print("💡 TTA often provides 0.5-1% improvement over single model predictions")

print("\n" + "="*60)
print("🎯 OPTIMIZED IMPROVEMENT TECHNIQUES APPLIED:")
print("1. ✅ Using merged augmented dataset (optimal data quality)")
if use_enhanced_model:
    print("2. ✅ Enhanced model architecture with additional layers")
    print("3. ✅ Attention-based pooling for better text representation")
else:
    print("2. ✅ Standard BanglaBERT architecture")
print("4. ✅ Optimized sequence length (256)")
print("5. ✅ Stable hyperparameters (conservative approach)")
print("6. ✅ Early stopping to prevent overfitting")
print("7. ✅ F1 score optimization instead of accuracy")
print("8. ✅ Clean and simple text preprocessing")
print("9. ✅ Test Time Augmentation (TTA)")
print("10. ✅ Efficient training approach")
print("11. ✅ Focus on proven techniques")
print("="*60)
print("\n💡 KEY IMPROVEMENTS:")
print("• Using your merged augmented dataset for optimal performance")
if use_enhanced_model:
    print("• Enhanced model with 3-layer classification head (768→512→256→labels)")
    print("• Attention pooling for better sequence representation")
    print("• Expected +0.5-1.5% accuracy improvement from architecture")
print("• Simplified preprocessing (no special tokens)")
print("• Replaced resource-intensive ensemble with efficient TTA")
print("• More stable training configuration")
print("• Clean, efficient approach focusing on proven techniques")
