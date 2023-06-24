import numpy as np
from datasets import load_dataset
from haxllm.dataset.hub import register_dataset


def load_fn(split, cache_dir=None):
    return load_dataset("HasturOfficial/adgen", split=split, cache_dir=cache_dir)


def preprocess_fn(tokenizer, example, max_len, max_source_length):
    max_target_length = max_len - max_source_length
    prompt_column = "content"
    response_column = "summary"
    n = len(example[prompt_column])
    inputs = np.full((n, max_len), tokenizer.pad_token_id, dtype=np.int32)
    labels = np.full((n, max_len), -100, dtype=np.int32)
    for i in range(n):
        prompt = example[prompt_column][i]
        answer = example[response_column][i]

        a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

        if len(a_ids) > max_source_length - 1:
            a_ids = a_ids[:max_source_length - 1]

        if len(b_ids) > max_target_length - 2:
            b_ids = b_ids[:max_target_length - 2]

        input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        l = len(input_ids)
        inputs[i, :l] = input_ids

        context_length = input_ids.index(tokenizer.bos_token_id)
        # shift left and pad the end with -100
        labels[i, context_length-1:l-2] = input_ids[context_length:-1]
    return {"input_ids": inputs, "labels": labels}
    

register_dataset("adgen", load_fn, preprocess_fn, ("train", "validation"))