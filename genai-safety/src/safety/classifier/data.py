"""Dataset loading for the classifier."""
import json
from typing import Optional


def load_splits(ccfg: dict, extra_train_path: Optional[str] = None):
    """Return a DatasetDict with 'train' and 'test', columns: text, label.

    extra_train_path is a JSONL file of {"text", "label"} rows (for example,
    auto-labeled or mined examples). It is added to TRAIN only, so the test
    split always stays human-labeled.
    """
    from datasets import Dataset, concatenate_datasets, load_dataset

    ds = load_dataset(ccfg["dataset"], revision=ccfg.get("dataset_revision"))
    keep = {ccfg["text_column"], ccfg["label_column"]}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    if ccfg["text_column"] != "text":
        ds = ds.rename_column(ccfg["text_column"], "text")
    if ccfg["label_column"] != "label":
        ds = ds.rename_column(ccfg["label_column"], "label")

    if extra_train_path:
        with open(extra_train_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if rows:
            extra = Dataset.from_list(
                [{"text": r["text"], "label": int(r["label"])} for r in rows])
            extra = extra.cast(ds["train"].features)
            ds["train"] = concatenate_datasets([ds["train"], extra]).shuffle(seed=0)
    return ds


def tokenize(ds, tok, max_length: int):
    return ds.map(lambda x: tok(x["text"], truncation=True, max_length=max_length),
                  batched=True)
