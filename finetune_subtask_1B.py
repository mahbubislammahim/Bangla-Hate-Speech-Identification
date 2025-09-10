# =============================================================================
# team_hate_mate
# BanglaBERT Hate Speech Detection
# Subtask 1B: Enhanced Model with Additional Layers
# =============================================================================

# Step 1: Install Dependencies
# =============================================================================
#!pip install transformers datasets evaluate accelerate huggingface_hub sentencepiece git+https://github.com/csebuetnlp/normalizer

# Step 2: Import Libraries
# =============================================================================
import logging
import os
import random
import sys
import unicodedata
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from normalizer import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import transformers
from transformers import (
    AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EvalPrediction, Trainer, TrainingArguments,
    default_data_collator, set_seed, EarlyStoppingCallback,
)

# Step 3: Setup Environment
# =============================================================================
os.environ["WANDB_DISABLED"] = "true"

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Step 4: Configuration
# =============================================================================
# Data files
train_file = 'blp25_hatespeech_subtask_1B_train.tsv'
validation_file = 'blp25_hatespeech_subtask_1B_dev.tsv'
test_file = 'blp25_hatespeech_subtask_1B_test.tsv'

model_name = 'csebuetnlp/banglabert'


max_seq_length = 256
use_enhanced_model = True   # Enhanced architecture on top of better base model

print(f"Using BanglaBERT model: {model_name}")

# Training settings - OPTIMIZED
training_args = TrainingArguments(
    learning_rate=4e-5,          
    num_train_epochs=3,          
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,  
    output_dir="./banglabert_improved_model/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    warmup_ratio=0.1,        
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    logging_steps=50,
    logging_dir="./logs",
    run_name="banglabert_1B_enhanced",
    report_to=None,
    dataloader_num_workers=0,
    fp16=True,
    save_steps=500,
    eval_steps=500,
)

set_seed(training_args.seed)


print("✅ Configuration loaded!")

# Step 5: Load Data
# =============================================================================
l2id = {'None': 0, 'Society': 1, 'Organization': 2, 'Community': 3, 'Individual': 4}

# Load datasets
print(f"📊 Loading datasets...")
train_df = pd.read_csv(train_file, sep='\t')
train_df['label'] = train_df['label'].map(l2id).fillna(0).astype(int)

validation_df = pd.read_csv(validation_file, sep='\t')
validation_df['label'] = validation_df['label'].map(l2id).fillna(0).astype(int)

test_df = pd.read_csv(test_file, sep='\t')

# Show statistics
print(f"   Training samples: {len(train_df):,}")
print(f"   Validation samples: {len(validation_df):,}")
print(f"   Test samples: {len(test_df):,}")

# Show class distribution
label_counts = train_df['label'].value_counts().sort_index()
weights = 1.0 / label_counts
weights = weights / weights.sum() * len(label_counts)
class_weights = torch.tensor(weights.values, dtype=torch.float).to("cuda")
print(class_weights)

id2l = {v: k for k, v in l2id.items()}
print("\n📊 Class Distribution:")
for label_id, count in label_counts.items():
    label_name = id2l[label_id]
    percentage = (count / len(train_df)) * 100
    print(f"   {label_name}: {count:,} ({percentage:.1f}%)")

# Convert to datasets
train_df = Dataset.from_pandas(train_df)
validation_df = Dataset.from_pandas(validation_df)
test_df = Dataset.from_pandas(test_df)

raw_datasets = DatasetDict({
    "train": train_df, 
    "validation": validation_df, 
    "test": test_df
})

print("✅ Data loaded successfully!")

# Step 6: Enhanced Model Architecture
# =============================================================================
class EnhancedBanglaBERT(nn.Module):
    """BanglaBERT with enhanced classification head"""
    
    def __init__(self, model_name, num_labels, use_attention_pooling=True, class_weights=None, label_smoothing: float = 0.0):
        super().__init__()
        self.num_labels = num_labels
        self.use_attention_pooling = use_attention_pooling
        self.label_smoothing = float(label_smoothing) if label_smoothing is not None else 0.0
        
        # Load pre-trained BanglaBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        
        # Update config for classification
        self.config.num_labels = num_labels
        self.config.label2id = {f"LABEL_{i}": i for i in range(num_labels)}
        self.config.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        
        if class_weights is not None:
            self.class_weights = class_weights
        
        hidden_size = self.bert.config.hidden_size
        
        # Attention pooling layer
        if use_attention_pooling:
            self.attention_pooling = nn.Linear(hidden_size, 1)
        
        # Optimized dropout configuration - progressive dropout
        self.dropout1 = nn.Dropout(0.1)  # Lower dropout after pooling
        self.dense1 = nn.Linear(hidden_size, 512)
        self.activation1 = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)  # Higher dropout before classifier
        self.classifier = nn.Linear(512, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.dense1, self.classifier]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Attention pooling or standard pooling
        if self.use_attention_pooling and attention_mask is not None:
            hidden_states = outputs.last_hidden_state
            attention_weights = self.attention_pooling(hidden_states).squeeze(-1)
            attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
            attention_weights = torch.softmax(attention_weights, dim=-1)
            pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        else:
            pooled_output = outputs.pooler_output
        
        # Enhanced classification head
        x = self.dropout1(pooled_output)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dropout2(x)
        
        logits = self.classifier(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            weights = getattr(self, 'class_weights', None)
            if weights is None:
                weights = torch.ones(self.num_labels).to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=self.label_smoothing)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        # Return only necessary outputs to avoid None values
        result = {
            "logits": logits,
        }
        
        if loss is not None:
            result["loss"] = loss
            
        return result

print("🧠 Enhanced model architecture defined!")

# Step 7: Load Model and Tokenizer
# =============================================================================
# Extract labels
label_list = raw_datasets["train"].unique("label")
label_list.sort()
num_labels = len(label_list)

# Load tokenizer
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

# Load model
if use_enhanced_model:
    print("🚀 Using Full Enhanced BanglaBERT...")
    model = EnhancedBanglaBERT(
        model_name=model_name,
        num_labels=num_labels,
        use_attention_pooling=True,
        label_smoothing=0.1,   # Optimal for hate speech classification
        # class_weights=class_weights # not using class weights
    )
    print(f"✅ Enhanced model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
else:
    print("📊 Using Standard BanglaBERT...")
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    print(f"✅ Standard model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Step 8: Preprocessing
# =============================================================================

def preprocess_function(examples):
    """Tokenize and clean text data"""
    texts = examples['text']
    
    # Clean text
    cleaned_texts = []
    for text in texts:
        text = str(text)
        text = normalize(text) # normalize data for Bangla
        cleaned_texts.append(text)
    
    # Tokenize
    result = tokenizer(
        cleaned_texts,
        padding=False,
        max_length=max_seq_length,
        truncation=True,
        return_tensors=None
    )
    
    # Handle labels
    if "label" in examples:
        result["label"] = examples["label"]
    
    return result

# Apply preprocessing
print("🔄 Preprocessing data...")
raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    desc="Tokenizing dataset",
)

# Remove text column
for split in raw_datasets.keys():
    if 'text' in raw_datasets[split].column_names:
        raw_datasets[split] = raw_datasets[split].remove_columns('text')

print(f"📊 Final columns: {raw_datasets['train'].column_names}")

# Step 9: Prepare Training
# =============================================================================
# Prepare datasets
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]
predict_dataset = raw_datasets["test"]

# Remove id column if exists
if "id" in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns("id")
if "id" in eval_dataset.column_names:
    eval_dataset = eval_dataset.remove_columns("id")

# Metrics function
def compute_metrics(p: EvalPrediction):
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



print("✅ Training preparation complete!")

# Step 10: Initialize Trainer
# =============================================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

print("✅ Trainer initialized!")

# Step 11: Train Model
# =============================================================================
print("🚀 Starting training...")
print(f"📊 Training samples: {len(train_dataset):,}")
print(f"📊 Validation samples: {len(eval_dataset):,}")
print(f"📊 Training for {training_args.num_train_epochs} epochs")

train_result = trainer.train()
metrics = train_result.metrics

print("✅ Training completed!")
print(f"📊 Training metrics: {metrics}")

# Step 12: Save Model
# =============================================================================
trainer.save_model()
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

print("✅ Model saved!")

# Step 13: Evaluate
# =============================================================================
print("📊 Evaluating model...")

# Define variables needed for evaluation
max_eval_samples = None
max_predict_samples = None

eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)

max_eval_samples = (
    max_eval_samples if max_eval_samples is not None else len(eval_dataset)
)
eval_metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

print(f"📊 Validation Accuracy: {eval_metrics.get('eval_accuracy', 'N/A'):.4f}")
print(f"📊 Validation F1: {eval_metrics.get('eval_f1', 'N/A'):.4f}")
print(f"📊 Validation Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")

# Confusion Matrix and Class-wise Report
print("\n🔎 Computing confusion matrix on validation set...")
eval_pred_output = trainer.predict(eval_dataset, metric_key_prefix="eval")
eval_preds = np.argmax(eval_pred_output.predictions, axis=1)
eval_labels = eval_pred_output.label_ids

labels_order = list(range(num_labels))
labels_names = [id2l[i] for i in labels_order]

cm = confusion_matrix(eval_labels, eval_preds, labels=labels_order)
cm_df = pd.DataFrame(cm, index=[f"true_{n}" for n in labels_names], columns=[f"pred_{n}" for n in labels_names])
print("\nConfusion Matrix (counts):")
print(cm_df.to_string())

cm_norm = (cm.T / np.maximum(cm.sum(axis=1), 1)).T
cm_norm_df = pd.DataFrame(np.round(cm_norm, 3), index=[f"true_{n}" for n in labels_names], columns=[f"pred_{n}" for n in labels_names])
print("\nConfusion Matrix (row-normalized = recall per class):")
print(cm_norm_df.to_string())

print("\nClass-wise precision/recall/F1:")
print(classification_report(eval_labels, eval_preds, labels=labels_order, target_names=labels_names, digits=3))

# Step 14: Generate Predictions
# =============================================================================
print("🎯 Generating predictions...")

# Get test predictions
ids = predict_dataset['id']
predict_dataset_clean = predict_dataset.remove_columns("id") if "id" in predict_dataset.column_names else predict_dataset
predictions = trainer.predict(predict_dataset_clean, metric_key_prefix="predict").predictions
predictions = np.argmax(predictions, axis=1)

# Save predictions
output_file = os.path.join(training_args.output_dir, "subtask_1B.tsv")
with open(output_file, "w") as writer:
    writer.write("id\tlabel\tmodel\n")
    for index, item in enumerate(predictions):
        item = label_list[item]
        item = id2l[item]
        writer.write(f"{ids[index]}\t{item}\tenhanced-{model_name}\n")

print(f"✅ Predictions saved to: {output_file}")

# Step 17: Summary
# =============================================================================
print("\n" + "="*60)
print("🎯 TRAINING COMPLETE!")
print("="*60)
if use_enhanced_model:
    print("Enhanced BanglaBERT with additional layers")
    print("   • Attention-based pooling")
else:
    print("Standard BanglaBERT model")

print(f"📊 Final Results:")
print(f"   • Validation Accuracy: {eval_metrics.get('eval_accuracy', 'N/A'):.4f}")
print(f"   • Validation F1: {eval_metrics.get('eval_f1', 'N/A'):.4f}")
print(f"   • Model saved in: {training_args.output_dir}")
print(f"   • Standard predictions: subtask_1B_predictions.tsv")
print("="*60)

# Step 18: Create Model Card (Optional)
# =============================================================================
# Create a model card for documentation
kwargs = {"finetuned_from": model_name, "tasks": "text-classification"}
trainer.create_model_card(**kwargs)

print("📄 Model card created!")
print("✅ All tasks completed successfully!")
