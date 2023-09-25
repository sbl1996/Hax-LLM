from typing import Optional

import numpy as np

from haxllm.chat.setting import ChatSetting


def find_prompt_prefix_suffix(examples, tokenizer, chat_setting):
    examples = [chat_setting.build_prompt(e) for e in examples]
    input_ids = [tokenizer.encode(e, add_special_tokens=True) for e in examples]
    l = 0
    while True:
        if len(set(input[l] for input in input_ids)) != 1:
            break
        l += 1
    r = 0
    while True:
        if len(set(input[-r-1] for input in input_ids)) != 1:
            break
        r += 1
    return input_ids[0][:l], input_ids[0][-r:]


# ChatGLM2 style dataset, modified from https://github.com/THUDM/ChatGLM2-6B/blob/main/ptuning/main.py#L158-L213
def preprocess_for_lm(
        tokenizer, examples, max_len, max_source_length, prompt_template=None,
        ignore_index=-100, padding_side=None,
        chat_setting: Optional[ChatSetting] = None, train=True):
    r"""
    Helper preprocess function for language modeling datasets.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used to tokenize the examples.
        The tokenizer is used for truncation but not padding.
        `padding_side` is regarded and padding is done by this function.
    examples : list of tuple
        List of examples. Each example is a tuple of two strings, representing the query and the answer.
    max_len : int
        Maximum length of the input sequence.
    max_source_length : int
        Maximum length of the query. max_len = max_source_length + max_target_length + 1
    prompt_template : str, optional
        Prompt template. <text> will be replaced with the query.
    ignore_index : int, default -100
        The index to ignore in the labels.
    padding_side : str, optional
        Padding side. If None, use tokenizer.padding_side.
    chat_setting : ChatSetting, optional
        Chat setting.
    train : bool, optional
        Whether to preprocess the training set or the validation set.
    """
    chat = chat_setting is not None
    if chat:
        prompt_prefix, prompt_suffix = find_prompt_prefix_suffix(["你好", "晚上吃什么", "请问你是谁", "How are you?"], tokenizer, chat_setting)
    if padding_side is None:
        padding_side = tokenizer.padding_side
    assert padding_side in ["left", "right"]

    query, answer = zip(*examples)
    n = len(query)

    if prompt_template:
        query = [prompt_template.replace("<text>", q) for q in query]
    if chat:
        query = [chat_setting.build_prompt(q) for q in query]

    if train:
        max_target_length = max_len - max_source_length - 1
        inputs = np.full((n, max_len), tokenizer.pad_token_id, dtype=np.int32)
        labels = np.full((n, max_len), ignore_index, dtype=np.int32)

        for i in range(n):
            a_ids = tokenizer.encode(text=query[i], add_special_tokens=True, truncation=True, max_length=max_source_length)

            # Option 1: a_ids + b_ids
            b_ids = tokenizer.encode(text=answer[i], add_special_tokens=False, truncation=True, max_length=max_target_length)

            if chat:
                # ensure that chat prompt is not truncated
                a_ids[-len(prompt_suffix):] = prompt_suffix
            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]

            # Option 2: encode(prompt + answer)
            # Note that a_ids + b_ids != encode(prompt + answer)
            # below is a workaround to fix this issue
            # input_ids = tokenizer.encode(text=prompt[i] + answer[i], add_special_tokens=True, truncation=True, max_length=max_source_length + max_target_length + 1)

            # context_length = len(a_ids)

            # if input_ids[:context_length] != a_ids:
            #     print(i)
            #     print("query: " + str(query[i]))
            #     print("answer: " + str(answer[i]))
            #     print("prompt: " + str(prompt[i]))

            #     print("input_ids[:context_length]: " + str(input_ids[:context_length]))
            #     print("a_ids: " + str(a_ids))
            #     raise ValueError("input_ids[:context_length] != a_ids")

            # input_ids[context_length - len(prompt_suffix):context_length] = prompt_suffix

            # input_ids.append(tokenizer.eos_token_id)
            
            # Option end

            l = len(input_ids)

            if padding_side == "right":
                inputs[i, :l] = input_ids
                
                # shift left and pad the end with ignore_index
                labels[i, context_length-1:l-1] = input_ids[context_length:]
            else:
                inputs[i, -l:] = input_ids
                
                # shift right and pad the beginning with ignore_index
                labels[i, -1-l+context_length:-1] = input_ids[context_length:]
        return {"input_ids": inputs, "labels": labels}
    else:
        max_target_length = max_len - max_source_length
        inputs = np.full((n, max_source_length), tokenizer.pad_token_id, dtype=np.int32)
        labels = np.full((n, max_target_length), tokenizer.pad_token_id, dtype=np.int32)

        encode_inputs = tokenizer(query, max_length=max_source_length, truncation=True)['input_ids']
        encode_labels = tokenizer(text_target=answer, add_special_tokens=False, max_length=max_target_length, truncation=True)['input_ids']

        for i in range(n):
            encode_input = encode_inputs[i]
            encode_label = encode_labels[i]
            if chat:
                encode_input[-len(prompt_suffix):] = prompt_suffix
            if padding_side == "right":
                inputs[i, :len(encode_input)] = encode_input
                labels[i, :len(encode_label)] = encode_label
            else:
                inputs[i, -len(encode_input):] = encode_input
                labels[i, -len(encode_label):] = encode_label
        return {"input_ids": inputs, "labels": labels}