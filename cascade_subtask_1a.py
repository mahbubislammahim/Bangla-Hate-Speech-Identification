import logging
import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from normalizer import normalize
from sklearn.metrics import f1_score, confusion_matrix, fbeta_score

# Disable W&B
# os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# File paths (same style as finetune_subtask_1B.py; expect files next to script or CWD)
TRAIN_1B = 'blp25_hatespeech_subtask_1A_train.tsv'
DEV_1B = 'blp25_hatespeech_subtask_1A_dev.tsv'
TEST_1B = 'blp25_hatespeech_subtask_1A_test.tsv'


# Label mappings
L2ID_1B = {'None': 0, 'Religious Hate': 1, 'Sexism': 2, 'Political Hate': 3, 'Profane': 4, 'Abusive': 5}

MODEL_NAME_STAGE1 = "csebuetnlp/banglabert"
MODEL_NAME_STAGE2 = "csebuetnlp/banglabert"

MAX_SEQ_LEN = 256


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


def build_tokenizer_and_model(model_name: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=None,
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,finetuning_task=None,
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
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
    
        eval_strategy="steps",
        eval_steps=200,
        logging_dir=output_dir,
        logging_strategy="steps",
        logging_steps=200,
        report_to=None,
    
        load_best_model_at_end=True,
        metric_for_best_model="eval_f2",
    
        fp16=True,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        lr_scheduler_type="linear",
    
        gradient_accumulation_steps=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    
        num_train_epochs=2
    )

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        acc = (preds == p.label_ids).astype(np.float32).mean().item()
        cm = confusion_matrix(p.label_ids, preds)
        print("Confusion matrix:\n", cm)
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

def train_stage_multi(dataset: DatasetDict, tokenizer, model, output_dir: str) -> Trainer:
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
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
    
        eval_strategy="steps",
        eval_steps=200,
        logging_dir=output_dir,
        logging_strategy="steps",
        logging_steps=200,
        report_to=None,
    
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
    
        fp16=True,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        lr_scheduler_type="linear",
    
        gradient_accumulation_steps=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    
        num_train_epochs=2
    )


    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        acc = (preds == p.label_ids).astype(np.float32).mean().item()
        cm = confusion_matrix(p.label_ids, preds)
        print("Confusion matrix:\n", cm)
        return {"accuracy": acc, "f1": f1_score(p.label_ids, preds, average="micro")}

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


def route_indices_by_stage1(pred_probs: np.ndarray, ids: List, threshold: float = 0.5):
    # pred_probs is probability for class 1 (hate)
    is_hate = pred_probs >= threshold
    hate_idx = np.where(is_hate)[0]
    none_idx = np.where(~is_hate)[0]
    return hate_idx, none_idx


def run_cascade(threshold: float = 0.5):
    # Stage 1: 1B -> binary (None vs Hate)
    logger.info("Loading 1B (binary) datasets...")
    ds1 = load_1b_binary_datasets()
    tok1, model1 = build_tokenizer_and_model(MODEL_NAME_STAGE1, num_labels=2)
    ds1 = preprocess_dataset(ds1, tok1, MAX_SEQ_LEN)
    trainer1 = train_stage_binary(ds1, tok1, model1, output_dir="./cascade_stage1_1B_binary")

    # Stage 2: 1B -> Multi (Again all other classes)
    logger.info("Loading 1B (multiclass) datasets...")
    ds2 = load_1b_multiclass_datasets()
    tok2, model2 = build_tokenizer_and_model(MODEL_NAME_STAGE2, num_labels=6)
    ds2 = preprocess_dataset(ds2, tok2, MAX_SEQ_LEN)

    # Filter Stage 2 training to hate-only labels (exclude None=0)
    train2 = ds2["train"]
    eval2 = ds2["validation"]
    test2 = ds2["test"]

    # train2_hate = train2.filter(lambda eg: eg["label"] != 0).map(lambda eg: {"label": eg["label"] - 1})
    # eval2_hate = eval2.filter(lambda eg: eg.get("label", 0) != 0).map(lambda eg: {"label": eg["label"] - 1})
    train2_hate = train2
    eval2_hate = eval2
    # Train stage 2 on hate-only
    stage2_train_ds = DatasetDict({"train": train2_hate, "validation": eval2_hate, "test": test2})
    trainer2 = train_stage_multi(stage2_train_ds, tok2, model2, output_dir="./cascade_stage2_1B_hate")

    # Cascaded inference on 1B dev and test
    id2l_1B = {v: k for k, v in L2ID_1B.items()}

    def predict_stage1(texts: List[str], batch_size: int = 32) -> np.ndarray:
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
                # Handle both Hugging Face model outputs (with .logits) and custom dict outputs
                if isinstance(outputs, dict):
                    logits = outputs['logits'].detach().cpu().numpy()
                else:
                    logits = outputs.logits.detach().cpu().numpy()
                probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
                all_probs.append(probs)
        
        return np.concatenate(all_probs, axis=0)

    def predict_stage2(texts: List[str], batch_size: int = 32) -> np.ndarray:
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
                # Handle both Hugging Face model outputs (with .logits) and custom dict outputs
                if isinstance(outputs, dict):
                    logits = outputs['logits'].detach().cpu().numpy()
                else:
                    logits = outputs.logits.detach().cpu().numpy()
                preds = np.argmax(logits, axis=1)
                all_preds.append(preds)
        
        return np.concatenate(all_preds, axis=0)

    # Dev routing
    dev_df_raw = pd.read_csv(DEV_1B, sep="\t")
    dev_ids = dev_df_raw["id"].tolist()
    dev_texts = dev_df_raw["text"].astype(str).tolist()
    dev_probs = predict_stage1(dev_texts)
    hate_idx, none_idx = route_indices_by_stage1(dev_probs, dev_ids, threshold)

    dev_final = np.zeros(len(dev_texts), dtype=int)
    dev_final[none_idx] = 0
    if len(hate_idx) > 0:
        sub_preds = predict_stage2([dev_texts[i] for i in hate_idx])
        dev_final[hate_idx] = sub_preds

    # Save dev predictions
    dev_out = os.path.join("./output", "subtask_1B_dev_cascade.tsv")
    os.makedirs(os.path.dirname(dev_out), exist_ok=True)
    with open(dev_out, "w", encoding="utf-8") as w:
        w.write("id\tlabel\tmodel\n")
        for i, pid in enumerate(dev_ids):
            w.write(f"{pid}\t{id2l_1B[int(dev_final[i])]}\tcascade(banglabert->banglabert)\n")
    logger.info(f"Saved cascaded dev predictions to {dev_out}")

    # Test routing
    test_df_raw = pd.read_csv(TEST_1B, sep="\t")
    test_ids = test_df_raw["id"].tolist()
    test_texts = test_df_raw["text"].astype(str).tolist()
    test_probs = predict_stage1(test_texts)
    hate_idx_t, none_idx_t = route_indices_by_stage1(test_probs, test_ids, threshold)

    test_final = np.zeros(len(test_texts), dtype=int)
    test_final[none_idx_t] = 0
    if len(hate_idx_t) > 0:
        sub_preds_t = predict_stage2([test_texts[i] for i in hate_idx_t])
        test_final[hate_idx_t] = sub_preds_t

    test_out = os.path.join("./output", "subtask_1B_test_cascade.tsv")
    with open(test_out, "w", encoding="utf-8") as w:
        w.write("id\tlabel\tmodel\n")
        for i, pid in enumerate(test_ids):
            w.write(f"{pid}\t{id2l_1B[int(test_final[i])]}\tcascade(banglabert->banglabert)\n")
    logger.info(f"Saved cascaded test predictions to {test_out}")


if __name__ == "__main__":
    run_cascade(threshold=0.3)
