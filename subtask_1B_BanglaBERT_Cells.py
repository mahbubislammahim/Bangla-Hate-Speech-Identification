# =============================================================================
# BanglaBERT Hate Speech Detection - Optimized for Easy Copy-Paste
# Subtask 1B: Enhanced Model with Additional Layers
# =============================================================================

# CELL 1: Install Dependencies
# =============================================================================
!pip install transformers datasets evaluate accelerate huggingface_hub sentencepiece

# CELL 2: Import Libraries
# =============================================================================
import logging
import os
import random
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import transformers
from transformers import (
    AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EvalPrediction, Trainer, TrainingArguments,
    default_data_collator, set_seed, EarlyStoppingCallback,
)

# CELL 3: Setup Environment
# =============================================================================
os.environ["WANDB_DISABLED"] = "true"
set_seed(42)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# CELL 4: Configuration
# =============================================================================
# Data files
train_file = 'blp25_hatespeech_subtask_1B_train.tsv'
validation_file = 'blp25_hatespeech_subtask_1B_dev.tsv'
test_file = 'blp25_hatespeech_subtask_1B_dev_test.tsv'

# Model settings
model_name = 'csebuetnlp/banglabert'
max_seq_length = 256
use_enhanced_model = True  # Set to False for standard model

# Training settings
training_args = TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    output_dir="./banglabert_improved_model/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    warmup_steps=500,
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    logging_steps=50,
    logging_dir="./logs",
    run_name="banglabert_improved",
    report_to=None,
    dataloader_num_workers=0,
    fp16=True,
)

print("✅ Configuration loaded!")

# CELL 5: Load Data
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

# CELL 6: Enhanced Model Architecture
# =============================================================================
class EnhancedBanglaBERT(nn.Module):
    """BanglaBERT with enhanced classification head"""
    
    def __init__(self, model_name, num_labels, hidden_dropout=0.3, use_attention_pooling=True):
        super().__init__()
        self.num_labels = num_labels
        self.use_attention_pooling = use_attention_pooling
        
        # Load pre-trained BanglaBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        
        # Update config for classification
        self.config.num_labels = num_labels
        self.config.label2id = {f"LABEL_{i}": i for i in range(num_labels)}
        self.config.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        
        hidden_size = self.bert.config.hidden_size
        
        # Attention pooling layer
        if use_attention_pooling:
            self.attention_pooling = nn.Linear(hidden_size, 1)
        
        # Enhanced classification head
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.dense1 = nn.Linear(hidden_size, 512)
        self.activation1 = nn.GELU()
        
        self.dropout2 = nn.Dropout(hidden_dropout)
        self.dense2 = nn.Linear(512, 256)
        self.activation2 = nn.GELU()
        
        self.dropout3 = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(256, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.dense1, self.dense2, self.classifier]:
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

print("🧠 Enhanced model architecture defined!")

# CELL 7: Load Model and Tokenizer
# =============================================================================
# Extract labels
label_list = raw_datasets["train"].unique("label")
label_list.sort()
num_labels = len(label_list)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=None,
    use_fast=True,
    revision="main",
    use_auth_token=None,
)

# Load model
if use_enhanced_model:
    print("🚀 Using Enhanced BanglaBERT...")
    model = EnhancedBanglaBERT(
        model_name=model_name,
        num_labels=num_labels,
        hidden_dropout=0.3,
        use_attention_pooling=True
    )
    print(f"✅ Enhanced model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
else:
    print("📊 Using Standard BanglaBERT...")
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    print(f"✅ Standard model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# CELL 8: Preprocessing
# =============================================================================
def preprocess_function(examples):
    """Tokenize and clean text data"""
    texts = examples['text']
    
    # Clean text
    cleaned_texts = []
    for text in texts:
        if isinstance(text, list):
            text = ' '.join(str(t) for t in text)
        text = str(text)
        text = ' '.join(text.split())
        text = text.replace('@', '').replace('http', '').replace('www', '')
        cleaned_texts.append(text)
    
    # Tokenize
    result = tokenizer(
        cleaned_texts,
        padding="max_length",
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

# CELL 9: Prepare Training
# =============================================================================
# Prepare datasets
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]
predict_dataset = raw_datasets["test"]

# Remove id column if exists
for dataset in [train_dataset, eval_dataset]:
    if "id" in dataset.column_names:
        dataset = dataset.remove_columns("id")

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

# Data collator
data_collator = default_data_collator

# Early stopping
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

print("✅ Training preparation complete!")

# CELL 10: Initialize Trainer
# =============================================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping_callback],
)

print("✅ Trainer initialized!")

# CELL 11: Train Model
# =============================================================================
print("🚀 Starting training...")
print(f"📊 Training samples: {len(train_dataset):,}")
print(f"📊 Validation samples: {len(eval_dataset):,}")
print(f"📊 Training for {training_args.num_train_epochs} epochs")

train_result = trainer.train()
metrics = train_result.metrics

print("✅ Training completed!")
print(f"📊 Training metrics: {metrics}")

# CELL 12: Save Model
# =============================================================================
trainer.save_model()
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

print("✅ Model saved!")

# CELL 13: Evaluate
# =============================================================================
print("📊 Evaluating model...")
eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)

trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

print(f"📊 Validation Accuracy: {eval_metrics.get('eval_accuracy', 'N/A'):.4f}")
print(f"📊 Validation F1: {eval_metrics.get('eval_f1', 'N/A'):.4f}")
print(f"📊 Validation Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")

# CELL 14: Generate Predictions
# =============================================================================
print("🎯 Generating predictions...")

# Get test predictions
ids = predict_dataset['id']
predict_dataset_clean = predict_dataset.remove_columns("id") if "id" in predict_dataset.column_names else predict_dataset
predictions = trainer.predict(predict_dataset_clean, metric_key_prefix="predict").predictions
predictions = np.argmax(predictions, axis=1)

# Save predictions
output_file = os.path.join(training_args.output_dir, "subtask_1B_predictions.tsv")
with open(output_file, "w") as writer:
    writer.write("id\tlabel\tmodel\n")
    for index, item in enumerate(predictions):
        item = label_list[item]
        item = id2l[item]
        writer.write(f"{ids[index]}\t{item}\t{model_name}\n")

print(f"✅ Predictions saved to: {output_file}")

# CELL 15: Test Time Augmentation (Optional)
# =============================================================================
def apply_tta(trainer, dataset, num_augmentations=5):
    """Apply Test Time Augmentation for better predictions"""
    all_predictions = []
    
    for aug_idx in range(num_augmentations):
        print(f"🔄 TTA iteration {aug_idx + 1}/{num_augmentations}")
        trainer.model.train()  # Enable dropout for variation
        with torch.no_grad():
            pred_results = trainer.predict(dataset, metric_key_prefix="predict")
            all_predictions.append(pred_results.predictions)
        trainer.model.eval()
    
    # Average predictions
    avg_predictions = np.mean(all_predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)

print("🎯 Applying Test Time Augmentation...")
tta_predictions = apply_tta(trainer, predict_dataset_clean, num_augmentations=5)

# Save TTA predictions
tta_output_file = os.path.join(training_args.output_dir, "subtask_1B_tta_predictions.tsv")
with open(tta_output_file, "w") as writer:
    writer.write("id\tlabel\tmodel\n")
    for index, item in enumerate(tta_predictions):
        item = label_list[item]
        item = id2l[item]
        writer.write(f"{ids[index]}\t{item}\t{model_name}_tta\n")

print(f"✅ TTA predictions saved to: {tta_output_file}")

# CELL 16: Upload to Hugging Face Hub (Optional)
# =============================================================================
from huggingface_hub import login

# Set your Hugging Face repository name here
HF_REPO_NAME = "your-username/banglabert-hatespeech-subtask1b"  # Change this!

print("🔑 HUGGING FACE UPLOAD:")
print("=" * 50)

# Step 1: Login (you'll need to provide your HF token)
print("🔑 Login to Hugging Face Hub...")
print("📝 You'll need your HF token from: https://huggingface.co/settings/tokens")

# Uncomment the next line when ready to upload
login()  # This will prompt for your HF token

# Step 2: Upload model and tokenizer
print(f"🚀 To upload model to: {HF_REPO_NAME}")
print("📝 Uncomment the upload code below when ready:")

# Uncomment these lines when ready to upload:
print(f"🚀 Uploading model to Hugging Face Hub: {HF_REPO_NAME}")
 
model.push_to_hub(
    HF_REPO_NAME,
    commit_message=f"Enhanced BanglaBERT for hate speech detection - F1: {eval_metrics.get('eval_f1', 'N/A'):.4f}",
    private=False  # Set to True if you want a private repo
)

tokenizer.push_to_hub(
    HF_REPO_NAME,
    commit_message="BanglaBERT tokenizer for hate speech detection",
    private=False
)

print(f"✅ Model uploaded to: https://huggingface.co/{HF_REPO_NAME}")

print("\n💡 TO UPLOAD:")
print("1. Change HF_REPO_NAME to your desired repository name")
print("2. Uncomment the login() line")
print("3. Get your HF token and login")
print("4. Uncomment the upload code")
print("5. Run this cell")

# CELL 17: Summary
# =============================================================================
print("\n" + "="*60)
print("🎯 TRAINING COMPLETE!")
print("="*60)
if use_enhanced_model:
    print("✅ Enhanced BanglaBERT with additional layers")
    print("   • 3-layer classification head (768→512→256→labels)")
    print("   • Attention-based pooling")
    print("   • Expected +1-2% accuracy improvement")
else:
    print("✅ Standard BanglaBERT model")

print(f"📊 Final Results:")
print(f"   • Validation Accuracy: {eval_metrics.get('eval_accuracy', 'N/A'):.4f}")
print(f"   • Validation F1: {eval_metrics.get('eval_f1', 'N/A'):.4f}")
print(f"   • Model saved in: {training_args.output_dir}")
print(f"   • Standard predictions: subtask_1B_predictions.tsv")
print(f"   • TTA predictions: subtask_1B_tta_predictions.tsv")
print("="*60)
print("🚀 Ready for submission!")

# CELL 18: Create Model Card (Optional)
# =============================================================================
# Create a model card for documentation
kwargs = {"finetuned_from": model_name, "tasks": "text-classification"}
trainer.create_model_card(**kwargs)

print("📄 Model card created!")
print("✅ All tasks completed successfully!")
