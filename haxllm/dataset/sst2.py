from typing import Sequence
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_data(split, tokenize_function, cache_dir=None):
    dataset = load_dataset('glue', 'sst2', split=split, cache_dir=cache_dir)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='numpy', columns=['input_ids', 'attention_mask', 'label']) # type: ignore
    input_ids = tokenized_dataset['input_ids'] # type: ignore
    attention_mask = tokenized_dataset['attention_mask'] # type: ignore
    labels = tokenized_dataset['label'] # type: ignore
    return input_ids, attention_mask, labels


def tokenize_function(tokenizer, example, max_len):
    return tokenizer(
        example['sentence'],
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='np'
    )
    

def create_dataset(tokenizer, max_len=128, eval_size=None, batch_size=128, eval_batch_size=None,
                   seed=42, with_test=False, sub_ratio=None, loader='tf', cache_dir=None, num_workers=0):
    assert eval_size is None, 'eval_size is not supported for SST-2'
    if eval_batch_size is None:
        eval_batch_size = batch_size

    tokenize_fn = lambda x: tokenize_function(tokenizer, x, max_len)
    if with_test:
        train_input_ids, train_attention_mask, train_labels = load_data('train', tokenize_fn, cache_dir=cache_dir)
        test_input_ids, test_attention_mask, test_labels = load_data('test', tokenize_fn, cache_dir=cache_dir)
    else:
        train_input_ids, train_attention_mask, train_labels = load_data('train', tokenize_fn, cache_dir=cache_dir)
        test_input_ids, test_attention_mask, test_labels = load_data('validation', tokenize_fn, cache_dir=cache_dir)

    if sub_ratio is not None:
        if isinstance(sub_ratio, Sequence):
            train_sub_ratio = sub_ratio[0]
            test_sub_ratio = sub_ratio[1]
        else:
            train_sub_ratio = sub_ratio
            test_sub_ratio = 1.0
        if train_sub_ratio < 1.0:
            train_input_ids, _, train_attention_mask, _, train_labels, _ = train_test_split(
                train_input_ids, train_attention_mask, train_labels, test_size=1-train_sub_ratio, random_state=seed)
        if test_sub_ratio < 1.0:
            test_input_ids, _, test_attention_mask, _, test_labels, _ = train_test_split(
                test_input_ids, test_attention_mask, test_labels, test_size=1-test_sub_ratio, random_state=seed)

    train_data = {'inputs': train_input_ids, 'attn_mask': train_attention_mask, 'labels': train_labels}
    test_data = {'inputs': test_input_ids, 'attn_mask': test_attention_mask, 'labels': test_labels}

    def cast_dtype(x):
        x['inputs'] = x['inputs'].astype(np.int32)
        x['attn_mask'] = x['attn_mask'].astype(np.bool_)
        return x

    train_data = cast_dtype(train_data)
    test_data = cast_dtype(test_data)

    if loader == 'tf':
        from haxllm.dataset.utils import create_tfds
        ds_train, steps_per_epoch = create_tfds(train_data, batch_size, train=True, seed=seed)
        ds_eval, eval_steps = create_tfds(test_data, eval_batch_size, train=False, seed=seed)
    elif loader == 'paddle':
        from haxllm.dataset.paddle.utils import create_paddle_loader
        ds_train, steps_per_epoch = create_paddle_loader(train_data, batch_size, train=True, num_workers=num_workers)
        ds_eval, eval_steps = create_paddle_loader(test_data, eval_batch_size, train=False, num_workers=num_workers)

    return ds_train, steps_per_epoch, ds_eval, eval_steps
