from typing import Sequence
import importlib

import numpy as np

from haxllm.gconfig import get_seed

dataset_hub = {}


def register_dataset(name, load_fn, preprocess_fn, splits, **options):
    if name in dataset_hub:
        raise ValueError(f"Dataset {name} already registered.")
    dataset_hub[name] = (load_fn, preprocess_fn, splits, options)


def get_registered_dataset(name):
    if name not in dataset_hub:
        try:
            importlib.import_module(f"haxllm.dataset.{name}")
        except ImportError:
            raise ValueError(f"Unknown dataset {name}")
    return dataset_hub[name]


def load_for_sequnece_classification(dataset, preprocess_fn):
    tokenized_dataset = dataset.map(preprocess_fn, batched=True)
    columns = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in tokenized_dataset.column_names:
        columns.append("token_type_ids")
    tokenized_dataset.set_format(type="numpy", columns=columns) # type: ignore
    input_ids = tokenized_dataset["input_ids"] # type: ignore
    attention_mask = tokenized_dataset["attention_mask"] # type: ignore
    labels = tokenized_dataset["label"] # type: ignore
    return {"inputs": input_ids, "attn_mask": attention_mask, "labels": labels}


def load_for_causal_lm(dataset, preprocess_fn):
    tokenized_dataset = dataset.map(preprocess_fn, batched=True)
    columns = ["input_ids", "labels"]
    tokenized_dataset.set_format(type="numpy", columns=columns) # type: ignore
    input_ids = tokenized_dataset["input_ids"] # type: ignore
    labels = tokenized_dataset["labels"] # type: ignore
    return {"input_ids": input_ids, "labels": labels}

def add_prefix(x, prefix):
    if isinstance(x, str):
        return f"{prefix} {x}"
    elif isinstance(x, Sequence):
        return [f"{prefix} {s}" for s in x]
    else:
        raise ValueError(f"Unknown type {type(x)}")

def create_dataset(
    name,
    tokenizer,
    max_len=128,
    splits=("train", "validation"),
    batch_size=128,
    seed=None,
    sub_ratio=None,
    task_type="sequnece_classification",
    loader="tf",
    cache_dir=None,
    num_workers=0,
    load_kwargs={},
    **kwargs,
):
    if seed is None:
        seed = get_seed()
    load_fn, preprocess_fn, ds_splits, options = get_registered_dataset(name)
    for split in splits:
        if split not in ds_splits:
            raise ValueError(f"Dataset {name} does not have split {split}.")

    assert len(splits) == 2, "Only support train/validation or train/test split for now."
    train_split, eval_split = splits

    if isinstance(batch_size, int):
        train_batch_size = eval_batch_size = batch_size
    else:
        train_batch_size, eval_batch_size = batch_size

    if sub_ratio is not None:
        if isinstance(sub_ratio, int):
            train_sub_ratio = eval_sub_ratio = sub_ratio
        else:
            train_sub_ratio, eval_sub_ratio = sub_ratio
        if train_sub_ratio == 1.0:
            train_sub_ratio = None
        if eval_sub_ratio == 1.0:
            eval_sub_ratio = None
    else:
        train_sub_ratio = eval_sub_ratio = None
    
    if task_type == "sequnece_classification":
        task_load_fn = load_for_sequnece_classification
    elif task_type == "causal_lm":
        task_load_fn = load_for_causal_lm
    else:
        raise ValueError(f"Unknown task type {task_type}")

    ds_train = load_fn(train_split, cache_dir=cache_dir, **load_kwargs)
    ds_train = ds_train.shuffle(seed=seed)
    if train_sub_ratio is not None:
        n_sub = int(len(ds_train) * train_sub_ratio)
        ds_train = ds_train.select(range(n_sub))
    preprocess_fn_train = lambda x: preprocess_fn(tokenizer, x, max_len, train=True, **kwargs)
    train_data = task_load_fn(ds_train, preprocess_fn_train)
    keys = list(train_data.keys())
    train_data = tuple(train_data[k] for k in keys)

    ds_eval = load_fn(eval_split, cache_dir=cache_dir, **load_kwargs)
    if eval_sub_ratio is not None:
        n_sub = int(len(ds_eval) * eval_sub_ratio)
        ds_eval = ds_eval.shuffle(seed=seed).select(range(n_sub))
    preprocess_fn_eval = lambda x: preprocess_fn(tokenizer, x, max_len, train=False, **kwargs)
    eval_data = task_load_fn(ds_eval, preprocess_fn_eval)
    eval_data = tuple(eval_data[k] for k in keys)

    train_data = dict(zip(keys, train_data))
    eval_data = dict(zip(keys, eval_data))

    def cast_dtype(x):
        if "inputs" in x:
            x["inputs"] = x["inputs"].astype(np.int32)
        if "input_ids" in x:
            x["input_ids"] = x["input_ids"].astype(np.int32)
        if "attn_mask" in x:
            x["attn_mask"] = x["attn_mask"].astype(np.bool_)
        return x

    train_data = cast_dtype(train_data)
    eval_data = cast_dtype(eval_data)

    if loader == "tf":
        from haxllm.dataset.utils import create_tfds
        ds_train, steps_per_epoch = create_tfds(train_data, train_batch_size, train=True, seed=seed)
        ds_eval, eval_steps = create_tfds(eval_data, eval_batch_size, train=False, seed=seed)
    elif loader == "paddle":
        from haxllm.dataset.paddle.utils import create_paddle_loader
        if isinstance(num_workers, Sequence):
            num_workers_train, num_workers_eval  = num_workers
        else:
            num_workers_train = num_workers_eval = num_workers
        ds_train, steps_per_epoch = create_paddle_loader(train_data, train_batch_size, train=True, num_workers=num_workers_train)
        ds_eval, eval_steps = create_paddle_loader(eval_data, eval_batch_size, train=False, num_workers=num_workers_eval)
    else:
        raise ValueError(f"Unknown loader {loader}.")

    return ds_train, steps_per_epoch, ds_eval, eval_steps
