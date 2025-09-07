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
from sklearn.metrics import f1_score

# Disable W&B
os.environ["WANDB_DISABLED"] = "true"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# File paths (same style as finetune_subtask_1B.py; expect files next to script or CWD)
TRAIN_1B = 'blp25_hatespeech_subtask_1B_train.tsv'
DEV_1B = 'blp25_hatespeech_subtask_1B_dev.tsv'
TEST_1B = 'blp25_hatespeech_subtask_1B_dev_test.tsv'


# Label mappings
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
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


def train_stage(dataset: DatasetDict, tokenizer, model, output_dir: str) -> Trainer:
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # Drop id for Trainer
    if "id" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("id")
    if "id" in eval_dataset.column_names:
        eval_dataset = eval_dataset.remove_columns("id")

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        warmup_ratio=0.08,
    )

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        acc = (preds == p.label_ids).astype(np.float32).mean().item()
        return {"accuracy": acc, "f1": f1_score(p.label_ids, preds, average="macro")}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
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
    trainer1 = train_stage(ds1, tok1, model1, output_dir="./cascade_stage1_1B_binary")

    # Stage 2: 1B -> 4 hate classes (trained on hate-only)
    logger.info("Loading 1B (multiclass) datasets...")
    ds2 = load_1b_multiclass_datasets()
    # Stage 2 will be trained on hate-only with labels remapped to 0..3
    tok2, model2 = build_tokenizer_and_model(MODEL_NAME_STAGE2, num_labels=4)
    ds2 = preprocess_dataset(ds2, tok2, MAX_SEQ_LEN)

    # Filter Stage 2 training to hate-only labels (exclude None=0)
    train2 = ds2["train"]
    eval2 = ds2["validation"]
    test2 = ds2["test"]

    train2_hate = train2.filter(lambda eg: eg["label"] != 0).map(lambda eg: {"label": eg["label"] - 1})
    eval2_hate = eval2.filter(lambda eg: eg.get("label", 0) != 0).map(lambda eg: {"label": eg["label"] - 1})

    # Train stage 2 on hate-only
    stage2_train_ds = DatasetDict({"train": train2_hate, "validation": eval2_hate, "test": test2})
    trainer2 = train_stage(stage2_train_ds, tok2, model2, output_dir="./cascade_stage2_1B_hate")

    # Cascaded inference on 1B dev and test
    id2l_1B = {v: k for k, v in L2ID_1B.items()}

    def predict_stage1(texts: List[str]) -> np.ndarray:
        enc = tok1([normalize(str(t)) for t in texts], padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        with torch.no_grad():
            outputs = trainer1.model(**{k: v.to(trainer1.model.device) for k, v in enc.items()})
            logits = outputs.logits.detach().cpu().numpy()
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        return probs

    def predict_stage2(texts: List[str]) -> np.ndarray:
        enc = tok2([normalize(str(t)) for t in texts], padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        with torch.no_grad():
            outputs = trainer2.model(**{k: v.to(trainer2.model.device) for k, v in enc.items()})
            logits = outputs.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
        return preds

    # Dev routing
    dev_df_raw = pd.read_csv(DEV_1B, sep="\t")
    dev_ids = dev_df_raw["id"].tolist()
    dev_texts = dev_df_raw["text"].astype(str).tolist()
    dev_probs = predict_stage1(dev_texts)
    hate_idx, none_idx = route_indices_by_stage1(dev_probs, dev_ids, threshold)

    dev_final = np.zeros(len(dev_texts), dtype=int)
    dev_final[none_idx] = 0
    if len(hate_idx) > 0:
        sub_preds = predict_stage2([dev_texts[i] for i in hate_idx])  # 0..3
        dev_final[hate_idx] = sub_preds + 1  # map back to 1..4

    # Save dev predictions
    dev_out = os.path.join("./cascade_outputs", "subtask_1B_dev_cascade.tsv")
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
        sub_preds_t = predict_stage2([test_texts[i] for i in hate_idx_t])  # 0 to 3
        test_final[hate_idx_t] = sub_preds_t + 1  # map back to 1..4

    test_out = os.path.join("./cascade_outputs", "subtask_1B_test_cascade.tsv")
    with open(test_out, "w", encoding="utf-8") as w:
        w.write("id\tlabel\tmodel\n")
        for i, pid in enumerate(test_ids):
            w.write(f"{pid}\t{id2l_1B[int(test_final[i])]}\tcascade(banglabert->banglabert)\n")
    logger.info(f"Saved cascaded test predictions to {test_out}")

    # Official evaluation on dev using provided scorer
    try:
        sys.path.append('.')
        from scorer.task import evaluate, _read_tsv_input_file, _read_gold_labels_file
        gold_file = DEV_1B
        pred_labels = _read_tsv_input_file(dev_out)
        gold_labels = _read_gold_labels_file(gold_file)
        acc, precision, recall, f1 = evaluate(pred_labels, gold_labels, '1B')
        logger.info(f"Official Dev Scores -> Acc: {acc:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
    except Exception as e:
        logger.warning(f"Evaluation skipped due to error: {e}")


if __name__ == "__main__":
    run_cascade(threshold=0.5)


