import numpy as np
from datasets import load_dataset
from haxllm.dataset.hub import register_dataset


def load_fn(split, cache_dir=None):
    return load_dataset("HasturOfficial/adgen", split=split, cache_dir=cache_dir)


def find_prompt_prefix_suffix(examples, tokenizer):
    examples = [tokenizer.build_prompt(e) for e in examples]
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
def preprocess_fn(tokenizer, example, max_len, max_source_length, prompt_template=None, train=True):
    prompt_column = "content"
    response_column = "summary"
    n = len(example[prompt_column])

    prompt_prefix, prompt_suffix = find_prompt_prefix_suffix(["你好", "晚上吃什么", "请问你是谁"], tokenizer)

    if train:
        max_target_length = max_len - max_source_length - 1
        inputs = np.full((n, max_len), tokenizer.pad_token_id, dtype=np.int32)
        labels = np.full((n, max_len), -100, dtype=np.int32)

        query = example[prompt_column]
        if prompt_template:
            query = [prompt_template.replace("<text>", q) for q in query]
        answer = example[response_column]
        prompt = [tokenizer.build_prompt(q) for q in query]

        for i in range(n):
            a_ids = tokenizer.encode(text=prompt[i], add_special_tokens=True, truncation=True, max_length=max_source_length)

            # Option 1: a_ids + b_ids
            b_ids = tokenizer.encode(text=answer[i], add_special_tokens=False, truncation=True, max_length=max_target_length)

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
            inputs[i, :l] = input_ids
            
            # shift left and pad the end with -100
            labels[i, context_length-1:l-1] = input_ids[context_length:]
        return {"input_ids": inputs, "labels": labels}
    else:
        max_target_length = max_len - max_source_length
        inputs = np.full((n, max_source_length), tokenizer.pad_token_id, dtype=np.int32)
        labels = np.full((n, max_target_length), -100, dtype=np.int32)

        query = example[prompt_column]
        answer = example[response_column]
        prompt = [tokenizer.build_prompt(q) for q in query]

        encode_inputs = tokenizer(prompt, max_length=max_source_length, truncation=True)['input_ids']
        encode_labels = tokenizer(text_target=answer, add_special_tokens=False, max_length=max_target_length, truncation=True)['input_ids']

        for i in range(n):
            encode_input = encode_inputs[i]
            encode_label = encode_labels[i]
            encode_input[-len(prompt_suffix):] = prompt_suffix
            inputs[i, :len(encode_input)] = encode_input
            labels[i, :len(encode_label)] = encode_label
        return {"input_ids": inputs, "labels": labels}

register_dataset("adgen", load_fn, preprocess_fn, ("train", "validation"))