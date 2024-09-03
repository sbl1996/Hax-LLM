from typing import List

from tqdm import tqdm

import numpy as np
from haxllm.pipeline.text_generation import TextGenerationPipeline


def prepare_input(tokenizer, inputs, max_len):
    n = len(inputs)
    padding_side = tokenizer.padding_side
    assert padding_side == "left"
    input_ids = np.full((n, max_len), tokenizer.pad_token_id, dtype=np.int32)
    encode_inputs = tokenizer(inputs, max_length=max_len, truncation=True)['input_ids']
    for i in range(n):
        encode_input = encode_inputs[i]
        input_ids[i, -len(encode_input):] = encode_input
    return input_ids


def batch_split(inputs, batch_size):
    batch_inputs = []
    mini_batch = []
    for s in inputs:
        mini_batch.append(s)
        if len(mini_batch) == batch_size:
            batch_inputs.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_inputs.append(mini_batch)
    return batch_inputs


def batch_inference(
    pipeline: TextGenerationPipeline,
    inputs: List[str],
    batch_size: int,
    max_new_tokens: int = 1,
    progress_bar: bool = False,
):
    tokenizer = pipeline.tokenizer
    pad_value = "pad"

    outputs = []
    total_samples = len(inputs)
    processed_samples = 0
    it = batch_split(inputs, batch_size)
    if progress_bar:
        pbar = tqdm(total=total_samples)
    for batch_input in it:
        pad_len = 0
        if len(batch_input) < batch_size:
            pad_len = batch_size - len(batch_input)
            batch_input = batch_input + [pad_value] * pad_len
        input_ids = prepare_input(tokenizer, batch_input, max_len=pipeline.max_len - max_new_tokens)
        output_ids = pipeline.random_sample(input_ids, max_new_tokens=max_new_tokens)[:(batch_size-pad_len)]
        outputs.extend(tokenizer.batch_decode(output_ids[:, -max_new_tokens:], skip_special_tokens=True))
        processed_samples += len(batch_input) - pad_len
        if progress_bar:
            pbar.update(len(batch_input) - pad_len)
    if progress_bar:
        pbar.close()
    return outputs