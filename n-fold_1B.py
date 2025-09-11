# =============================================================================
# Enhanced 5-Fold Cross-Validation Cascade Implementation for Subtask 1B
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
from sklearn.model_selection import StratifiedKFold

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
    
    def __init__(self, model_name: str, num_labels: int, hidden_dropout: float = 0.2, use_attention_pooling: bool = True,  label_smoothing: float = 0.0):
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
        
        hidden_size = self.bert.config.hidden_size
        
        # Attention pooling layer
        if use_attention_pooling:
            self.attention_pooling = nn.Linear(hidden_size, 1)
        
        # Enhanced classification head
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(hidden_size, 512)
        self.activation1 = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)
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
            loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
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
        hidden_dropout=0.2,
        use_attention_pooling=True,
        label_smoothing=0.1
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


def train_stage_binary_cv(dataset: DatasetDict, tokenizer_factory, model_factory, base_output_dir: str, n_folds: int = 5) -> List[Trainer]:
    """Train 5-fold cross-validation models for stage 1 binary classification"""
    train_dataset = dataset["train"]
    
    # Convert to pandas for stratified splitting
    train_df = train_dataset.to_pandas()
    
    # Setup stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    trainers = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        logger.info(f"Training binary stage fold {fold + 1}/{n_folds}")
        
        # Split data for this fold
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Convert back to datasets
        fold_train_dataset = Dataset.from_pandas(fold_train_df)
        fold_val_dataset = Dataset.from_pandas(fold_val_df)
        
        # Drop id columns for trainer
        if "id" in fold_train_dataset.column_names:
            fold_train_dataset = fold_train_dataset.remove_columns("id")
        if "id" in fold_val_dataset.column_names:
            fold_val_dataset = fold_val_dataset.remove_columns("id")
        
        # Create fresh model and tokenizer for this fold
        tokenizer = tokenizer_factory()
        model = model_factory()
        
        # Setup output directory for this fold
        fold_output_dir = os.path.join(base_output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        args = TrainingArguments(
            learning_rate=3e-5,
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            output_dir=fold_output_dir,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f2",
            greater_is_better=True,
            warmup_ratio=0.1,
            weight_decay=0.01,
            gradient_accumulation_steps=2,
            logging_steps=50,
            logging_dir=os.path.join(fold_output_dir, "logs"),
            run_name=f"fold_{fold}",
            report_to=None,
            dataloader_num_workers=0,
            fp16=True,
            lr_scheduler_type="linear",
            label_smoothing_factor=0.1
        )

        def compute_metrics(p):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            acc = (preds == p.label_ids).astype(np.float32).mean().item()
            cm = confusion_matrix(p.label_ids, preds)
            print(f"Binary fold {fold + 1} confusion matrix:\n", cm)
            return {"accuracy": acc, "f2": fbeta_score(p.label_ids, preds, beta=2, average="micro")}

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_val_dataset,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model()
        trainers.append(trainer)
        
        logger.info(f"Completed binary fold {fold + 1}/{n_folds}")
    
    return trainers


def train_stage_enhanced_multiclass_cv(dataset: DatasetDict, tokenizer_factory, model_factory, base_output_dir: str, n_folds: int = 5) -> List[Trainer]:
    """Train 5-fold cross-validation models for stage 2 enhanced multiclass classification"""
    train_dataset = dataset["train"]
    
    # Convert to pandas for stratified splitting
    train_df = train_dataset.to_pandas()
    
    # Setup stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    trainers = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        logger.info(f"Training enhanced multiclass stage fold {fold + 1}/{n_folds}")
        
        # Split data for this fold
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Convert back to datasets
        fold_train_dataset = Dataset.from_pandas(fold_train_df)
        fold_val_dataset = Dataset.from_pandas(fold_val_df)
        
        # Drop id columns for trainer
        if "id" in fold_train_dataset.column_names:
            fold_train_dataset = fold_train_dataset.remove_columns("id")
        if "id" in fold_val_dataset.column_names:
            fold_val_dataset = fold_val_dataset.remove_columns("id")
        
        # Create fresh model and tokenizer for this fold
        tokenizer = tokenizer_factory()
        model = model_factory()
        
        # Setup output directory for this fold
        fold_output_dir = os.path.join(base_output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        args = TrainingArguments(
            output_dir=fold_output_dir,
            overwrite_output_dir=True,
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="epoch",
            logging_dir=fold_output_dir,
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
            print(f"Enhanced multiclass fold {fold + 1} confusion matrix:\n", cm)
            return {"accuracy": acc, "f1": f1_score(p.label_ids, preds, average="micro")}

        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_val_dataset,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping_callback],
        )

        trainer.train()
        trainer.save_model()
        trainers.append(trainer)
        
        logger.info(f"Completed enhanced multiclass fold {fold + 1}/{n_folds}")
    
    return trainers


def predict_stage1_ensemble(trainers: List[Trainer], tokenizers: List, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Ensemble prediction across CV folds for stage 1 by averaging probabilities"""
    n_folds = len(trainers)
    all_fold_probs = []
    
    for fold_idx, (trainer, tokenizer) in enumerate(zip(trainers, tokenizers)):
        logger.info(f"Stage 1 prediction with fold {fold_idx + 1}/{n_folds}")
        
        fold_probs = []
        model = trainer.model
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
                probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]  # probability for class 1 (hate)
                fold_probs.append(probs)
        
        fold_probs = np.concatenate(fold_probs, axis=0)
        all_fold_probs.append(fold_probs)
    
    # Average probabilities across folds
    ensemble_probs = np.mean(all_fold_probs, axis=0)
    return ensemble_probs


def predict_stage2_ensemble(trainers: List[Trainer], tokenizers: List, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Ensemble prediction across CV folds for stage 2 by averaging predictions"""
    n_folds = len(trainers)
    all_fold_preds = []
    
    for fold_idx, (trainer, tokenizer) in enumerate(zip(trainers, tokenizers)):
        logger.info(f"Stage 2 prediction with fold {fold_idx + 1}/{n_folds}")
        
        fold_preds = []
        model = trainer.model
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
                fold_preds.append(preds)
        
        fold_preds = np.concatenate(fold_preds, axis=0)
        all_fold_preds.append(fold_preds)
    
    # Take mode (most common prediction) across folds
    all_fold_preds = np.array(all_fold_preds)  # shape: (n_folds, n_samples)
    ensemble_preds = []
    for i in range(all_fold_preds.shape[1]):
        # Get predictions for sample i across all folds
        sample_preds = all_fold_preds[:, i]
        # Take the most common prediction
        unique, counts = np.unique(sample_preds, return_counts=True)
        ensemble_pred = unique[np.argmax(counts)]
        ensemble_preds.append(ensemble_pred)
    
    return np.array(ensemble_preds)


def route_indices_by_stage1(pred_probs: np.ndarray, ids: List, threshold: float = 0.5):
    """Route samples based on stage 1 predictions"""
    # pred_probs is probability for class 1 (hate)
    is_hate = pred_probs >= threshold
    hate_idx = np.where(is_hate)[0]
    none_idx = np.where(~is_hate)[0]
    return hate_idx, none_idx


def run_enhanced_cascade_cv(threshold: float = 0.3):
    """Run the enhanced cascade approach with 5-fold cross-validation"""
    logger.info("Starting Enhanced 5-Fold CV Cascade for Subtask 1B")
    
    # Stage 1: Binary classification (None vs Hate) with 5-fold CV
    logger.info("=== STAGE 1: Binary Classification with 5-Fold CV ===")
    logger.info("Loading 1B (binary) datasets...")
    ds1 = load_1b_binary_datasets()
    
    # Factory functions for creating fresh models/tokenizers for each fold
    def binary_tokenizer_factory():
        tokenizer, _ = build_standard_tokenizer_and_model(MODEL_NAME_STAGE1, num_labels=2)
        return tokenizer
    
    def binary_model_factory():
        _, model = build_standard_tokenizer_and_model(MODEL_NAME_STAGE1, num_labels=2)
        return model
    
    # Get a tokenizer for preprocessing (will be used for all folds)
    tok1, _ = build_standard_tokenizer_and_model(MODEL_NAME_STAGE1, num_labels=2)
    ds1 = preprocess_dataset(ds1, tok1, MAX_SEQ_LEN)
    
    # Train 5-fold CV models for stage 1
    logger.info("Training 5-fold CV models for stage 1 binary classification...")
    cv_trainers_binary = train_stage_binary_cv(ds1, binary_tokenizer_factory, binary_model_factory, 
                                             base_output_dir="./enhanced_cascade_stage1_1B_binary_cv", n_folds=5)
    
    # Create tokenizers for each fold for inference
    cv_tokenizers_binary = [binary_tokenizer_factory() for _ in range(5)]

    # Stage 2: Enhanced multiclass classification (All 5 classes) with 5-fold CV
    logger.info("=== STAGE 2: Enhanced Multiclass Classification with 5-Fold CV ===")
    logger.info("Loading 1B (multiclass) datasets...")
    ds2 = load_1b_multiclass_datasets()
    
    # Factory functions for enhanced models
    def enhanced_tokenizer_factory():
        tokenizer, _ = build_enhanced_tokenizer_and_model(MODEL_NAME_STAGE2, num_labels=5)
        return tokenizer
    
    def enhanced_model_factory():
        _, model = build_enhanced_tokenizer_and_model(MODEL_NAME_STAGE2, num_labels=5)
        return model
    
    # Get a tokenizer for preprocessing
    tok2, _ = build_enhanced_tokenizer_and_model(MODEL_NAME_STAGE2, num_labels=5)
    ds2 = preprocess_dataset(ds2, tok2, MAX_SEQ_LEN)
    
    # Train 5-fold CV enhanced models for stage 2
    logger.info("Training 5-fold CV models for stage 2 enhanced multiclass classification...")
    cv_trainers_enhanced = train_stage_enhanced_multiclass_cv(ds2, enhanced_tokenizer_factory, enhanced_model_factory,
                                                            base_output_dir="./enhanced_cascade_stage2_1B_multiclass_cv", n_folds=5)
    
    # Create tokenizers for each fold for inference
    cv_tokenizers_enhanced = [enhanced_tokenizer_factory() for _ in range(5)]

    # Cascaded inference on 1B dev and test
    id2l_1B = {v: k for k, v in L2ID_1B.items()}

    # Dev routing with ensemble prediction
    logger.info("=== INFERENCE ON DEV SET ===")
    dev_df_raw = pd.read_csv(DEV_1B, sep="\t")
    dev_ids = dev_df_raw["id"].tolist()
    dev_texts = dev_df_raw["text"].astype(str).tolist()
    
    # Stage 1 ensemble predictions
    logger.info("Performing ensemble prediction for stage 1 on dev set...")
    dev_probs = predict_stage1_ensemble(cv_trainers_binary, cv_tokenizers_binary, dev_texts)
    hate_idx, none_idx = route_indices_by_stage1(dev_probs, dev_ids, threshold)
    
    logger.info(f"Stage 1 routing: {len(none_idx)} samples -> None, {len(hate_idx)} samples -> Stage 2")

    dev_final = np.zeros(len(dev_texts), dtype=int)
    dev_final[none_idx] = 0  # Assign None directly
    
    if len(hate_idx) > 0:
        # Stage 2 ensemble predictions for hate samples
        logger.info("Performing ensemble prediction for stage 2 on dev set...")
        sub_preds = predict_stage2_ensemble(cv_trainers_enhanced, cv_tokenizers_enhanced, 
                                          [dev_texts[i] for i in hate_idx])
        dev_final[hate_idx] = sub_preds

    # Save dev predictions
    dev_out = os.path.join("./enhanced_cascade_cv_outputs", "subtask_1B_dev_enhanced_cascade_cv.tsv")
    os.makedirs(os.path.dirname(dev_out), exist_ok=True)
    with open(dev_out, "w", encoding="utf-8") as w:
        w.write("id\tlabel\tmodel\n")
        for i, pid in enumerate(dev_ids):
            w.write(f"{pid}\t{id2l_1B[int(dev_final[i])]}\tenhanced_cascade_cv(banglabert->enhanced_banglabert)\n")
    logger.info(f"Saved enhanced cascaded CV dev predictions to {dev_out}")

    # Test routing with ensemble prediction
    logger.info("=== INFERENCE ON TEST SET ===")
    test_df_raw = pd.read_csv(TEST_1B, sep="\t")
    test_ids = test_df_raw["id"].tolist()
    test_texts = test_df_raw["text"].astype(str).tolist()
    
    # Stage 1 ensemble predictions
    logger.info("Performing ensemble prediction for stage 1 on test set...")
    test_probs = predict_stage1_ensemble(cv_trainers_binary, cv_tokenizers_binary, test_texts)
    hate_idx_t, none_idx_t = route_indices_by_stage1(test_probs, test_ids, threshold)
    
    logger.info(f"Stage 1 routing: {len(none_idx_t)} samples -> None, {len(hate_idx_t)} samples -> Stage 2")

    test_final = np.zeros(len(test_texts), dtype=int)
    test_final[none_idx_t] = 0  # Assign None directly
    
    if len(hate_idx_t) > 0:
        # Stage 2 ensemble predictions for hate samples
        logger.info("Performing ensemble prediction for stage 2 on test set...")
        sub_preds_t = predict_stage2_ensemble(cv_trainers_enhanced, cv_tokenizers_enhanced,
                                            [test_texts[i] for i in hate_idx_t])
        test_final[hate_idx_t] = sub_preds_t

    test_out = os.path.join("./enhanced_cascade_cv_outputs", "subtask_1B_test_enhanced_cascade_cv.tsv")
    with open(test_out, "w", encoding="utf-8") as w:
        w.write("id\tlabel\tmodel\n")
        for i, pid in enumerate(test_ids):
            w.write(f"{pid}\t{id2l_1B[int(test_final[i])]}\tenhanced_cascade_cv(banglabert->enhanced_banglabert)\n")
    logger.info(f"Saved enhanced cascaded CV test predictions to {test_out}")
    
    logger.info("Enhanced 5-fold CV cascade completed successfully!")


if __name__ == "__main__":
    run_enhanced_cascade_cv(threshold=0.3)
