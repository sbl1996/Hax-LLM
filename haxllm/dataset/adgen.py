from datasets import load_dataset
from haxllm.dataset.lm_base import preprocess_for_lm
from haxllm.dataset.hub import register_dataset


def load_fn(split, cache_dir=None):
    return load_dataset("HasturOfficial/adgen", split=split, cache_dir=cache_dir)


def preprocess_fn(tokenizer, examples, max_len, train=True, **kwargs):
    prompt_column = "content"
    response_column = "summary"
    responses = [x.replace("<UNK>", "") for x in examples[response_column]]
    examples = list(zip(examples[prompt_column], responses))
    return preprocess_for_lm(tokenizer, examples, max_len=max_len, train=train, **kwargs)

register_dataset("adgen", load_fn, preprocess_fn, ("train", "validation"))