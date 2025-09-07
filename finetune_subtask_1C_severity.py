# =============================================================================
# team_hate_mate
# BanglaBERT Hate Speech Detection
# Subtask 1C (severity-only): Fine-tune to predict only hate_severity
# Mirrors enhanced 1B setup for strong single-label performance
# =============================================================================

# Step 1: Imports and environment
# =============================================================================
import logging
import os
import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from normalizer import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import transformers
from transformers import (
    AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
    EvalPrediction, Trainer, TrainingArguments,
    default_data_collator, set_seed, EarlyStoppingCallback,
)

os.environ["WANDB_DISABLED"] = "true"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Step 2: Data files and label mapping
# =============================================================================
train_file = 'blp25_hatespeech_subtask_1C_train.tsv'
validation_file = 'blp25_hatespeech_subtask_1C_dev.tsv'
test_file = 'blp25_hatespeech_subtask_1C_dev_test.tsv'

model_name = 'csebuetnlp/banglabert'
max_seq_length = 256
use_enhanced_model = False 

# Severity label mapping from 1C
severity_l2id = {'Little to None': 0, 'Mild': 1, 'Severe': 2}
id2severity = {v: k for k, v in severity_l2id.items()}


# Step 3: Training configuration (aligned with strong 1B settings)
# =============================================================================
training_args = TrainingArguments(
    learning_rate=4e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    output_dir="./banglabert_1C_severity/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    warmup_steps=300,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    logging_steps=50,
    logging_dir="./logs",
    run_name="banglabert_1C_severity",
    report_to=None,
    dataloader_num_workers=0,
    fp16=True,
    lr_scheduler_type="linear",
    save_steps=500,
    eval_steps=500,
)

set_seed(training_args.seed)


# Step 4: Load and prepare datasets
# =============================================================================
logger.info("Loading 1C datasets and mapping severity → label...")

train_df = pd.read_csv(train_file, sep='\t')
validation_df = pd.read_csv(validation_file, sep='\t')
test_df = pd.read_csv(test_file, sep='\t')

train_df['label'] = train_df['hate_severity'].map(severity_l2id).fillna(0).astype(int)
validation_df['label'] = validation_df['hate_severity'].map(severity_l2id).fillna(0).astype(int)

print(f"   Training samples: {len(train_df):,}")
print(f"   Validation samples: {len(validation_df):,}")
print(f"   Test samples: {len(test_df):,}")

# Show class distribution
label_counts = train_df['label'].value_counts().sort_index()
id2l = {0: 'Little to None', 1: 'Mild', 2: 'Severe'}
print("\n📊 Severity Class Distribution (train):")
for label_id, count in label_counts.items():
    label_name = id2l[label_id]
    percentage = (count / len(train_df)) * 100
    print(f"   {label_name}: {count:,} ({percentage:.1f}%)")

# Convert to datasets
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(validation_df)
test_ds = Dataset.from_pandas(test_df)

raw_datasets = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds,
})


# Step 5: Define enhanced model head (mirrors 1B)
# =============================================================================
class EnhancedBanglaBERT(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dropout=0.3, use_attention_pooling=True):
        super().__init__()
        self.num_labels = num_labels
        self.use_attention_pooling = use_attention_pooling
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        if use_attention_pooling:
            self.attention_pooling = nn.Linear(hidden_size, 1)
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.dense1 = nn.Linear(hidden_size, 512)
        self.activation1 = nn.GELU()
        self.dropout2 = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(512, num_labels)
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
            return_dict=True,
        )

        if self.use_attention_pooling and attention_mask is not None:
            hidden_states = outputs.last_hidden_state
            attention_weights = self.attention_pooling(hidden_states).squeeze(-1)
            attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
            attention_weights = torch.softmax(attention_weights, dim=-1)
            pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        else:
            pooled_output = outputs.pooler_output

        x = self.dropout1(pooled_output)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dropout2(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        result = {"logits": logits}
        if loss is not None:
            result["loss"] = loss
        return result


# Step 6: Tokenizer and preprocessing
# =============================================================================
label_list = sorted(raw_datasets["train"].unique("label"))
num_labels = len(label_list)

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

def preprocess_function(examples):
    texts = examples['text']
    cleaned_texts = [normalize(str(t)) for t in texts]
    result = tokenizer(
        cleaned_texts,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        return_tensors=None,
    )
    if "label" in examples:
        result["label"] = examples["label"]
    return result

raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    desc="Tokenizing dataset",
)

for split in raw_datasets.keys():
    if 'text' in raw_datasets[split].column_names:
        raw_datasets[split] = raw_datasets[split].remove_columns('text')


# Step 7: Model, metrics, trainer
# =============================================================================
if use_enhanced_model:
    print("🚀 Using Enhanced head for 1C severity...")
    model = EnhancedBanglaBERT(
        model_name=model_name,
        num_labels=num_labels,
        hidden_dropout=0.3,
        use_attention_pooling=True,
    )
else:
    print("📊 Using AutoModelForSequenceClassification for 1C severity...")
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    accuracy = (preds == p.label_ids).astype(np.float32).mean().item()
    f1_micro = f1_score(p.label_ids, preds, average='micro')
    precision_w = precision_score(p.label_ids, preds, average='weighted')
    recall_w = recall_score(p.label_ids, preds, average='weighted')
    return {
        "accuracy": accuracy,
        "f1": f1_micro,
        "precision": precision_w,
        "recall": recall_w,
    }

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]
predict_dataset = raw_datasets["test"]

# Remove id column if exists (to avoid passing unexpected kwargs to model)
if "id" in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns("id")
if "id" in eval_dataset.column_names:
    eval_dataset = eval_dataset.remove_columns("id")

data_collator = default_data_collator
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping_callback],
)


# Step 8: Train and save
# =============================================================================
print("🚀 Starting training (1C severity-only)...")
train_result = trainer.train()
metrics = train_result.metrics
trainer.save_model()
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


# Step 9: Evaluate and predict
# =============================================================================
eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

print("🎯 Generating severity predictions...")
ids = predict_dataset['id'] if 'id' in predict_dataset.column_names else list(range(len(predict_dataset)))
predict_dataset_clean = predict_dataset.remove_columns("id") if "id" in predict_dataset.column_names else predict_dataset
pred_logits = trainer.predict(predict_dataset_clean, metric_key_prefix="predict").predictions
pred_indices = np.argmax(pred_logits, axis=1)

output_file = os.path.join(training_args.output_dir, "subtask_1C_severity.tsv")
with open(output_file, "w") as writer:
    writer.write("id\thate_severity\tmodel\n")
    for index, pred_idx in enumerate(pred_indices):
        label_id = label_list[pred_idx]
        label_name = id2severity[label_id]
        writer.write(f"{ids[index]}\t{label_name}\t{model_name}\n")

print(f"✅ Predictions saved to: {output_file}")


# Step 10: Done
# =============================================================================
print("\n" + "="*60)
print("🎯 1C Severity-only training COMPLETE!")
print("="*60)
print(f"   • Validation F1: {eval_metrics.get('eval_f1', 'N/A'):.4f}")
print(f"   • Model saved in: {training_args.output_dir}")
print(f"   • Predictions: subtask_1C_severity.tsv")
print("="*60)


