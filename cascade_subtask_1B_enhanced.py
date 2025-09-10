# =============================================================================
# Enhanced Cascade Implementation for Subtask 1B
# =============================================================================

import logging
import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
from normalizer import normalize
from sklearn.metrics import f1_score, confusion_matrix, fbeta_score

# Disable W&B
os.environ["WANDB_MODE"] = "disabled"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# File paths for subtask 1B
TRAIN_1B = 'blp25_hatespeech_subtask_1B_train.tsv'
DEV_1B = 'blp25_hatespeech_subtask_1B_dev.tsv'
TEST_1B = 'blp25_hatespeech_subtask_1B_test.tsv'

# Label mappings for subtask 1B
L2ID_1B = {
    "None": 0,
    "Society": 1,
    "Organization": 2,
    "Community": 3,
    "Individual": 4,
}

MODEL_NAME_STAGE1 = "csebuetnlp/banglabert"
MODEL_NAME_STAGE2 = "csebuetnlp/banglabert"

MAX_SEQ_LEN = 256


class EnhancedBanglaBERT(nn.Module):
    """Enhanced BanglaBERT with additional layers for better classification"""
    
    def __init__(self, model_name: str, num_labels: int, hidden_dropout: float = 0.3, use_attention_pooling: bool = True):
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
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        result = {"logits": logits}
        if loss is not None:
            result["loss"] = loss
            
        return result


def load_1b_binary_datasets() -> DatasetDict:
    """Load 1B data and convert to binary labels: 0 -> None, 1 -> Hate (any non-None)."""
    train_df = pd.read_csv(TRAIN_1B, sep="\t")
    dev_df = pd.read_csv(DEV_1B, sep="\t")
    test_df = pd.read_csv(TEST_1B, sep="\t")

    def to_binary(df: pd.DataFrame) -> pd.DataFrame:
        mapped = df.copy()
        mapped["label"] = df["label"].map(L2ID_1B).fillna(0).astype(int).apply(lambda x: 0 if x == 0 else 1)
        return mapped

    train_df = to_binary(train_df)
    dev_df = to_binary(dev_df)
    # test has no labels; keep as-is

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(dev_df),
            "test": Dataset.from_pandas(test_df),
        }
    )


def load_1b_multiclass_datasets() -> DatasetDict:
    """Load 1B data; keep original 5 labels with 0 as None."""
    train_df = pd.read_csv(TRAIN_1B, sep="\t")
    dev_df = pd.read_csv(DEV_1B, sep="\t")
    test_df = pd.read_csv(TEST_1B, sep="\t")

    for df in (train_df, dev_df):
        df["label"] = df["label"].map(L2ID_1B).fillna(0).astype(int)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(dev_df),
            "test": Dataset.from_pandas(test_df),
        }
    )


def build_standard_tokenizer_and_model(model_name: str, num_labels: int):
    """Build standard model for binary classification stage"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=None,
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task=None,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        from_tf=bool(".ckpt" in model_name),
        cache_dir=None,
        revision="main",
        use_auth_token=None,
        ignore_mismatched_sizes=False,
    )
    return tokenizer, model


def build_enhanced_tokenizer_and_model(model_name: str, num_labels: int):
    """Build enhanced model for multiclass classification stage"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EnhancedBanglaBERT(
        model_name=model_name,
        num_labels=num_labels,
        hidden_dropout=0.3,
        use_attention_pooling=True
    )
    return tokenizer, model


def preprocess_dataset(ds: DatasetDict, tokenizer, max_len: int) -> DatasetDict:
    def preprocess(examples):
        texts = [normalize(str(t)) for t in examples["text"]]
        result = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len)
        if "label" in examples:
            result["label"] = examples["label"]
        return result

    processed = DatasetDict()
    for split, split_ds in ds.items():
        # keep id for later routing; we will drop on Trainer input when needed
        processed[split] = split_ds.map(preprocess, batched=True, desc=f"Tokenizing {split}")
    return processed


def train_stage_binary(dataset: DatasetDict, tokenizer, model, output_dir: str) -> Trainer:
    """Train binary classification stage (None vs Hate)"""
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # Drop id for Trainer
    if "id" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("id")
    if "id" in eval_dataset.column_names:
        eval_dataset = eval_dataset.remove_columns("id")

    args = TrainingArguments(
        learning_rate=3e-5,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        output_dir=output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f2",
        greater_is_better=True,
        warmup_ratio=0.08,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        logging_steps=50,
        logging_dir=os.path.join(output_dir, "logs"),
        run_name=os.path.basename(output_dir),
        report_to=None,
        dataloader_num_workers=0,
        fp16=True,
        lr_scheduler_type="linear",
        save_steps=500,
        eval_steps=500,
        label_smoothing_factor=0.1
    )

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        acc = (preds == p.label_ids).astype(np.float32).mean().item()
        cm = confusion_matrix(p.label_ids, preds)
        print("Binary stage confusion matrix:\n", cm)
        return {"accuracy": acc, "f2": fbeta_score(p.label_ids, preds, beta=2, average="micro")}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    return trainer


def train_stage_enhanced_multiclass(dataset: DatasetDict, tokenizer, model, output_dir: str) -> Trainer:
    """Train enhanced multiclass classification stage"""
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # Drop id for Trainer
    if "id" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("id")
    if "id" in eval_dataset.column_names:
        eval_dataset = eval_dataset.remove_columns("id")

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        save_strategy="epoch",
        save_total_limit=2,
    
        eval_strategy="epoch",
        logging_dir=output_dir,
        logging_strategy="steps",
        logging_steps=50,
        report_to=None,
    
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    
        fp16=True,
        learning_rate=4e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
    
        gradient_accumulation_steps=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
    
        num_train_epochs=3
    )

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        acc = (preds == p.label_ids).astype(np.float32).mean().item()
        cm = confusion_matrix(p.label_ids, preds)
        print("Enhanced multiclass stage confusion matrix:\n", cm)
        return {"accuracy": acc, "f1": f1_score(p.label_ids, preds, average="micro")}

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    trainer.save_model()
    return trainer


def route_indices_by_stage1(pred_probs: np.ndarray, ids: List, threshold: float = 0.5):
    """Route samples based on stage 1 predictions"""
    # pred_probs is probability for class 1 (hate)
    is_hate = pred_probs >= threshold
    hate_idx = np.where(is_hate)[0]
    none_idx = np.where(~is_hate)[0]
    return hate_idx, none_idx


def run_enhanced_cascade(threshold: float = 0.3):
    """Run the enhanced cascade approach"""
    logger.info("Starting Enhanced Cascade for Subtask 1B")
    
    # Stage 1: Binary classification (None vs Hate) with standard model
    logger.info("=== STAGE 1: Binary Classification ===")
    logger.info("Loading 1B (binary) datasets...")
    ds1 = load_1b_binary_datasets()
    tok1, model1 = build_standard_tokenizer_and_model(MODEL_NAME_STAGE1, num_labels=2)
    ds1 = preprocess_dataset(ds1, tok1, MAX_SEQ_LEN)
    trainer1 = train_stage_binary(ds1, tok1, model1, output_dir="./enhanced_cascade_stage1_1B_binary")

    # Stage 2: Enhanced multiclass classification (All 5 classes including None)
    logger.info("=== STAGE 2: Enhanced Multiclass Classification ===")
    logger.info("Loading 1B (multiclass) datasets...")
    ds2 = load_1b_multiclass_datasets()
    tok2, model2 = build_enhanced_tokenizer_and_model(MODEL_NAME_STAGE2, num_labels=5)
    ds2 = preprocess_dataset(ds2, tok2, MAX_SEQ_LEN)
    
    # Train enhanced model on all classes
    trainer2 = train_stage_enhanced_multiclass(ds2, tok2, model2, output_dir="./enhanced_cascade_stage2_1B_multiclass")

    # Cascaded inference on 1B dev and test
    id2l_1B = {v: k for k, v in L2ID_1B.items()}

    def predict_stage1(texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Predict using stage 1 (binary) model"""
        all_probs = []
        model = trainer1.model
        tokenizer = tok1
        device = next(model.parameters()).device

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            normalized_texts = [normalize(str(t)) for t in batch_texts]
            enc = tokenizer(normalized_texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
                if isinstance(outputs, dict):
                    logits = outputs['logits'].detach().cpu().numpy()
                else:
                    logits = outputs.logits.detach().cpu().numpy()
                probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
                all_probs.append(probs)
        
        return np.concatenate(all_probs, axis=0)

    def predict_stage2(texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Predict using stage 2 (enhanced multiclass) model"""
        all_preds = []
        model = trainer2.model
        tokenizer = tok2
        device = next(model.parameters()).device

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            normalized_texts = [normalize(str(t)) for t in batch_texts]
            enc = tokenizer(normalized_texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
                if isinstance(outputs, dict):
                    logits = outputs['logits'].detach().cpu().numpy()
                else:
                    logits = outputs.logits.detach().cpu().numpy()
                preds = np.argmax(logits, axis=1)
                all_preds.append(preds)
        
        return np.concatenate(all_preds, axis=0)

    # Dev routing
    logger.info("=== INFERENCE ON DEV SET ===")
    dev_df_raw = pd.read_csv(DEV_1B, sep="\t")
    dev_ids = dev_df_raw["id"].tolist()
    dev_texts = dev_df_raw["text"].astype(str).tolist()
    
    # Stage 1 predictions
    dev_probs = predict_stage1(dev_texts)
    hate_idx, none_idx = route_indices_by_stage1(dev_probs, dev_ids, threshold)
    
    logger.info(f"Stage 1 routing: {len(none_idx)} samples -> None, {len(hate_idx)} samples -> Stage 2")

    dev_final = np.zeros(len(dev_texts), dtype=int)
    dev_final[none_idx] = 0  # Assign None directly
    
    if len(hate_idx) > 0:
        # Stage 2 predictions for hate samples
        sub_preds = predict_stage2([dev_texts[i] for i in hate_idx])
        dev_final[hate_idx] = sub_preds

    # Save dev predictions
    dev_out = os.path.join("./enhanced_cascade_outputs", "subtask_1B_dev_enhanced_cascade.tsv")
    os.makedirs(os.path.dirname(dev_out), exist_ok=True)
    with open(dev_out, "w", encoding="utf-8") as w:
        w.write("id\tlabel\tmodel\n")
        for i, pid in enumerate(dev_ids):
            w.write(f"{pid}\t{id2l_1B[int(dev_final[i])]}\tenhanced_cascade(banglabert->enhanced_banglabert)\n")
    logger.info(f"Saved enhanced cascaded dev predictions to {dev_out}")

    # Test routing
    logger.info("=== INFERENCE ON TEST SET ===")
    test_df_raw = pd.read_csv(TEST_1B, sep="\t")
    test_ids = test_df_raw["id"].tolist()
    test_texts = test_df_raw["text"].astype(str).tolist()
    
    # Stage 1 predictions
    test_probs = predict_stage1(test_texts)
    hate_idx_t, none_idx_t = route_indices_by_stage1(test_probs, test_ids, threshold)
    
    logger.info(f"Stage 1 routing: {len(none_idx_t)} samples -> None, {len(hate_idx_t)} samples -> Stage 2")

    test_final = np.zeros(len(test_texts), dtype=int)
    test_final[none_idx_t] = 0  # Assign None directly
    
    if len(hate_idx_t) > 0:
        # Stage 2 predictions for hate samples
        sub_preds_t = predict_stage2([test_texts[i] for i in hate_idx_t])
        test_final[hate_idx_t] = sub_preds_t

    test_out = os.path.join("./enhanced_cascade_outputs", "subtask_1B_test_enhanced_cascade.tsv")
    with open(test_out, "w", encoding="utf-8") as w:
        w.write("id\tlabel\tmodel\n")
        for i, pid in enumerate(test_ids):
            w.write(f"{pid}\t{id2l_1B[int(test_final[i])]}\tenhanced_cascade(banglabert->enhanced_banglabert)\n")
    logger.info(f"Saved enhanced cascaded test predictions to {test_out}")
    
    logger.info("Enhanced cascade completed successfully!")


if __name__ == "__main__":
    run_enhanced_cascade(threshold=0.3)
