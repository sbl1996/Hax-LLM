import numpy as np
from datasets import load_dataset
from haxllm.dataset.hub import register_dataset


def load_fn(split, cache_dir=None):
    return load_dataset("boolq", split=split, cache_dir=cache_dir)


def preprocess_fn(tokenizer, example, max_len, passage=True):
    if passage:
        sentences = list(zip(example["passage"], example["question"]))
    else:
        sentences = example["question"]

    d = tokenizer(
        sentences,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="np"
    )
    d["label"] = np.array(example["answer"], dtype=np.int64)
    return d
    

register_dataset("boolq", load_fn, preprocess_fn, ("train", "validation"))