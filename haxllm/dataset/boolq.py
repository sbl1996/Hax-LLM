import numpy as np
from datasets import load_dataset
from haxllm.dataset.hub import register_dataset


def load_fn(split, cache_dir=None):
    return load_dataset('boolq', split=split, cache_dir=cache_dir)


def preprocess_fn(tokenizer, example, max_len):
    d = tokenizer(
        example['sentence'],
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='np'
    )
    d['label'] = np.array(example['answer'], dtype=np.int64)
    return d
    

register_dataset('sst2', load_fn, preprocess_fn, ('train', 'validation'))