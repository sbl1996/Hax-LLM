from datasets import load_dataset
from haxllm.dataset.hub import register_dataset


def load_fn(split, cache_dir):
    return load_dataset("imdb", split=split, cache_dir=cache_dir)


def preprocess_fn(tokenizer, example, max_len, train=True):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="np"
    )


register_dataset("imdb", load_fn, preprocess_fn, ("train", "validation", "test"),
                 split_train_for_validation=True, split_ratio=0.2)