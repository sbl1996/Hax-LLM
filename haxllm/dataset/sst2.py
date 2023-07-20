from datasets import load_dataset
from haxllm.dataset.hub import register_dataset


def load_fn(split, cache_dir=None):
    return load_dataset("glue", "sst2", split=split, cache_dir=cache_dir)


def preprocess_fn(tokenizer, example, max_len, train=True):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="np"
    )


register_dataset("sst2", load_fn, preprocess_fn, ("train", "validation"))