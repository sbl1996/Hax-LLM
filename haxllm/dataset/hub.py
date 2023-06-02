from typing import Sequence
import importlib

import numpy as np
from sklearn.model_selection import train_test_split


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
    tokenized_dataset.set_format(type='numpy', columns=['input_ids', 'attention_mask', 'label']) # type: ignore
    input_ids = tokenized_dataset['input_ids'] # type: ignore
    attention_mask = tokenized_dataset['attention_mask'] # type: ignore
    labels = tokenized_dataset['label'] # type: ignore
    return input_ids, attention_mask, labels


def create_dataset(
    name,
    tokenizer,
    max_len=128,
    splits=('train', 'validation'),
    batch_size=128,
    seed=42,
    sub_ratio=None,
    loader="tf",
    cache_dir=None,
    num_workers=0,
):
    load_fn, preprocess_fn_, ds_splits, options = get_registered_dataset(name)
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
    
    preprocess_fn = lambda x: preprocess_fn_(tokenizer, x, max_len)
    ds_train = load_fn(train_split, cache_dir=cache_dir)
    train_input_ids, train_attention_mask, train_labels = load_for_sequnece_classification(ds_train, preprocess_fn)

    if eval_split == 'validation' and options.get('split_train_for_validation', False):
        test_size = options.get('split_ratio', 0.2)
        train_input_ids, eval_input_ids, train_attention_mask, eval_attention_mask, train_labels, eval_labels = train_test_split(
            train_input_ids, train_attention_mask, train_labels, test_size=test_size, random_state=seed)
    else:
        ds_eval = load_fn(eval_split, cache_dir=cache_dir)
        eval_input_ids, eval_attention_mask, eval_labels = load_for_sequnece_classification(ds_eval, preprocess_fn)

    # shuffle train data first
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_input_ids))
    train_input_ids = train_input_ids[perm]
    train_attention_mask = train_attention_mask[perm]
    train_labels = train_labels[perm]

    if train_sub_ratio is not None:
        train_input_ids, _, train_attention_mask, _, train_labels, _ = train_test_split(
            train_input_ids, train_attention_mask, train_labels, test_size=1-train_sub_ratio, random_state=seed)
    if eval_sub_ratio is not None:
        eval_input_ids, _, eval_attention_mask, _, eval_labels, _ = train_test_split(
            eval_input_ids, eval_attention_mask, eval_labels, eval_size=1-eval_sub_ratio, random_state=seed)

    train_data = {'inputs': train_input_ids, 'attn_mask': train_attention_mask, 'labels': train_labels}
    eval_data = {'inputs': eval_input_ids, 'attn_mask': eval_attention_mask, 'labels': eval_labels}

    def cast_dtype(x):
        x['inputs'] = x['inputs'].astype(np.int32)
        x['attn_mask'] = x['attn_mask'].astype(np.bool_)
        return x

    train_data = cast_dtype(train_data)
    eval_data = cast_dtype(eval_data)

    if loader == 'tf':
        from haxllm.dataset.utils import create_tfds
        ds_train, steps_per_epoch = create_tfds(train_data, train_batch_size, train=True, seed=seed)
        ds_eval, eval_steps = create_tfds(eval_data, eval_batch_size, train=False, seed=seed)
    elif loader == 'paddle':
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
