# =============================================================================
# BanglaBERT Hate Speech Detection - Optimized for Easy Copy-Paste
# Subtask 1B: Enhanced Model with Additional Layers
# =============================================================================

# CELL 1: Install Dependencies
# =============================================================================
#pip install transformers datasets evaluate accelerate huggingface_hub sentencepiece

# CELL 2: Import Libraries
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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

# Model settings - BETTER BASE MODEL
model_name = 'csebuetnlp/banglabert_large'  
# model_name='FacebookAI/xlm-roberta-base'

max_seq_length = 256
use_enhanced_model = True   # Enhanced architecture on top of better base model

print(f"🚀 Using LARGE BanglaBERT model: {model_name}")
print("💡 Large models typically perform 3-5% better than base models")

# Training settings - OPTIMIZED for Large Model
training_args = TrainingArguments(
    learning_rate=2e-5,          
    num_train_epochs=5,          
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,  
    output_dir="./banglabert_improved_model/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    warmup_ratio=0.06,        
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    logging_steps=50,
    logging_dir="./logs",
    run_name="banglabert_balanced",
    report_to=None,
    dataloader_num_workers=2,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    label_smoothing_factor=0.05,
    group_by_length=True,
    eval_accumulation_steps=2,
    optim="adamw_torch_fused",
    max_grad_norm=0.5,
    # Conservative optimizations
    lr_scheduler_type="linear",   # More stable than cosine
    save_steps=500,
    eval_steps=500,
)

print("✅ Configuration loaded!")

# CELL 5: Load Data
# =============================================================================
l2id = {'None': 0, 'Society': 1, 'Organization': 2, 'Community': 3, 'Individual': 4}

# Load datasets
print(f"📊 Loading datasets...")
train_df = pd.read_csv(train_file, sep='\t')
mapped = train_df['label'].map(l2id)
unknown_train = mapped.isna().sum()
if unknown_train > 0:
    print(f"⚠️  {unknown_train} unknown train labels mapped to 'None'.")
mapped = mapped.fillna(l2id['None'])
train_df['label'] = mapped.astype(int)

validation_df = pd.read_csv(validation_file, sep='\t')
mapped_val = validation_df['label'].map(l2id)
unknown_val = mapped_val.isna().sum()
if unknown_val > 0:
    print(f"⚠️  {unknown_val} unknown val labels mapped to 'None'.")
mapped_val = mapped_val.fillna(l2id['None'])
validation_df['label'] = mapped_val.astype(int)

test_df = pd.read_csv(test_file, sep='\t')

# Show statistics
print(f"   Training samples: {len(train_df):,}")
print(f"   Validation samples: {len(validation_df):,}")
print(f"   Test samples: {len(test_df):,}")

# Show class distribution
label_counts = train_df['label'].value_counts().sort_index()

# Calculate class weights with oversampling consideration
# Weight for each class is proportional to the majority class count
majority_class_count = label_counts.max()
oversampling_weights = majority_class_count / label_counts

# Normalize weights to be suitable for CrossEntropyLoss
# This step ensures the scale of the weights is manageable
oversampling_weights = oversampling_weights / oversampling_weights.sum() * len(label_counts)

class_weights = torch.tensor(oversampling_weights.values, dtype=torch.float)
print("Class weights with oversampling:")
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

# CELL 6: Enhanced Model Architecture
# =============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Args:
        gamma (float): focusing parameter; larger values focus more on hard examples.
        alpha (Tensor|List|None): class weights (optional). Shape [num_classes].
        reduction (str): 'mean' | 'sum' | 'none'.
        label_smoothing (float): label smoothing factor in [0, 1). Applied before focal modulation.
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is None:
            self.register_buffer('alpha', None)
        else:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float)
            self.register_buffer('alpha', alpha_tensor)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # One-hot targets with optional label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0)
            if self.label_smoothing > 0.0:
                true_dist = true_dist * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        # Compute focal modulation
        pt = (true_dist * probs).sum(dim=-1)
        focal_factor = (1.0 - pt).clamp(min=1e-6).pow(self.gamma)

        # Weighted negative log-likelihood
        nll = -(true_dist * log_probs).sum(dim=-1)

        if self.alpha is not None:
            alpha_weight = self.alpha[target]
            loss = alpha_weight * focal_factor * nll
        else:
            loss = focal_factor * nll

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

def create_focal_loss(class_weights: torch.Tensor = None, gamma: float = 2.0, label_smoothing: float = 0.0) -> FocalLoss:
    """Helper to build FocalLoss with optional class weights and smoothing.

    Note: This only constructs the loss; do not call here. Use later in training.
    """
    alpha = class_weights if class_weights is not None else None
    return FocalLoss(gamma=gamma, alpha=alpha, reduction='mean', label_smoothing=label_smoothing)

class EnhancedBanglaBERT(nn.Module):
    """BanglaBERT with enhanced classification head"""
    
    def __init__(self, model_name, num_labels, hidden_dropout=0.2, use_attention_pooling=True, class_weights=None):
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
        
        if class_weights is not None:
            # Keep with model and move across devices automatically
            self.register_buffer("class_weights", class_weights)
        
        hidden_size = self.bert.config.hidden_size
        
        # Attention pooling layer
        if use_attention_pooling:
            self.attention_pooling = nn.Linear(hidden_size, 1)
        
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.dense1 = nn.Linear(hidden_size, 512)
        self.activation1 = nn.GELU()
        self.dropout2 = nn.Dropout(hidden_dropout)
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
            # Default: class-weighted CrossEntropy with mild smoothing
            loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

            # --- FocalLoss alternative (commented out) ---
            # If you want to use FocalLoss instead of CrossEntropyLoss, uncomment below:
            # if not hasattr(self, 'focal_loss'):
            #     self.focal_loss = create_focal_loss(
            #         class_weights=weights,
            #         gamma=2.0,
            #         label_smoothing=0.02,
            #     )
            # loss = self.focal_loss(logits, labels)
            # -------------------------------------------
        
        # Return only necessary outputs to avoid None values
        result = {
            "logits": logits,
        }
        
        if loss is not None:
            result["loss"] = loss
            
        return result

print("🧠 Enhanced model architecture defined!")

# CELL 7: Load Model and Tokenizer
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
        hidden_dropout=0.3,
        use_attention_pooling=True,
        class_weights=class_weights,
    )
    print("🧠 Architecture: Large BERT → Attention Pool → Dense(512) → Classifier")
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
        # Normalize unicode and common Bangla artifacts
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\u200c', '').replace('\u200d', '')  # zero-width non/joiners
        text = text.replace('@', '').replace('http', '').replace('www', '')
        text = ' '.join(text.split())
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

# CELL 9: Prepare Training
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
    f1_macro = f1_score(p.label_ids, preds, average='macro', zero_division=0)
    f1_micro = f1_score(p.label_ids, preds, average='micro', zero_division=0)
    f1_weighted = f1_score(p.label_ids, preds, average='weighted', zero_division=0)
    precision = precision_score(p.label_ids, preds, average='weighted', zero_division=0)
    recall = recall_score(p.label_ids, preds, average='weighted', zero_division=0)
    
    # Calculate per-class F1 scores
    per_class_f1 = f1_score(p.label_ids, preds, average=None, labels=list(range(num_labels)), zero_division=0)
    
    # Print per-class F1 scores
    print("\n--- Per-Class F1 Scores ---")
    for i, score in enumerate(per_class_f1):
        label_name = id2l.get(i, f"Class_{i}")
        print(f"   {label_name}: {score:.4f}")
    print("--------------------------\n")
    
    return {
        "accuracy": accuracy,
        "f1": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall
    }

# Data collator - dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

# Early stopping
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0005)

print("✅ Training preparation complete!")

# CELL 10: Initialize Trainer
# =============================================================================
class CustomTrainer(Trainer):
    """Trainer with WeightedRandomSampler for class imbalance"""
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")

        labels = self.train_dataset["label"]
        labels = np.array(labels)
        class_counts = np.bincount(labels, minlength=num_labels)
        # Avoid division by zero
        class_counts[class_counts == 0] = 1
        sample_weights = 1.0 / class_counts[labels]
        sample_weights = torch.from_numpy(sample_weights).float()

        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
        )

trainer = CustomTrainer(
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
tta_predictions = apply_tta(trainer, predict_dataset_clean, num_augmentations=10)

# Save TTA predictions
tta_output_file = os.path.join(training_args.output_dir, "subtask_1B_tta_predictions.tsv")
with open(tta_output_file, "w") as writer:
    writer.write("id\tlabel\tmodel\n")
    for index, item in enumerate(tta_predictions):
        item = label_list[item]
        item = id2l[item]
        writer.write(f"{ids[index]}\t{item}\t{model_name}_tta\n")

print(f"✅ TTA predictions saved to: {tta_output_file}")

# CELL 17: Summary
# =============================================================================
print("\n" + "="*60)
print("🎯 TRAINING COMPLETE!")
print("="*60)
if use_enhanced_model:
    print("✅ Enhanced BanglaBERT with additional layers")
    print("   • 2-layer classification head (768→512→labels)")
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
