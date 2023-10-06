import os
import time 

import json
from glob import glob

from tqdm import tqdm

import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig

from haxllm.chat.config_utils import load_config
from haxllm.gconfig import set_gconfig


choices = ["A", "B", "C", "D"]


def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def prepare_input(tokenizer, prompts, max_len):
    n = len(prompts)
    padding_side = tokenizer.padding_side
    inputs = np.full((n, max_len), tokenizer.pad_token_id, dtype=np.int32)

    encode_inputs = tokenizer(prompts, max_length=max_len, truncation=True)['input_ids']

    for i in range(n):
        encode_input = encode_inputs[i]
        if padding_side == "right":
            inputs[i, :len(encode_input)] = encode_input
        else:
            inputs[i, -len(encode_input):] = encode_input
    return inputs


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(pipeline, tokenizer, prompts, batch_size):
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        pad_len = 0
        if len(batch_input) < batch_size:
            pad_len = batch_size - len(batch_input)
            batch_input = batch_input + ['Pad'] * pad_len
        input_ids = prepare_input(tokenizer, batch_input, max_len=pipeline.max_len - 1)
        output_ids = pipeline.random_sample(input_ids, max_new_tokens=1)[:(batch_size-pad_len)]
        answers.extend(tokenizer.batch_decode(output_ids[:, -1], skip_special_tokens=True))
    return answers


def load(cfg):
    pipeline = load_config(cfg, chat=False)[0]
    tokenizer = pipeline.tokenizer
    return pipeline, tokenizer


@hydra.main(version_base=None, config_path="../configs/chat", config_name="base")
def main(cfg: DictConfig) -> None:
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.initialize_cache(os.path.expanduser("~/jax_cache"))

    set_gconfig({
        "seed": cfg.seed,
    })

    required = ["max_len", "batch_size", "data_dir"]
    for r in required:
        assert getattr(cfg, r, None) is not None, "%s must be specified in config" % r
    
    data_dir = cfg.data_dir
    assert os.path.exists(data_dir), "data_dir %s does not exist" % data_dir

    start_time = time.time()

    run_results = {}
    k = getattr(cfg, "shot", 5)
    output_filename = 'run_results_%s.json' % cfg.model

    # compute_metric(output_filename)
    # raise NotImplementedError

    suffix = "_dev.csv"

    tasks = sorted(list(map(lambda x: x.split("/")[-1][:-len(suffix)], glob(os.path.join(data_dir, "dev", "*" + suffix)))))

    print("Use left padding for batch inference")
    cfg.padding_side = "left"
    pipeline, tokenizer = load(cfg)
    print(pipeline.model.config)

    for task in tasks:
        print('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(data_dir, "dev", task + "_dev.csv"), header=None)[:k]
        train_prompt = gen_prompt(dev_df, task, k)
        test_df = pd.read_csv(os.path.join(data_dir, "test", task + "_test.csv"), header=None)
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            prompt_end = format_example(test_df, i, include_answer=False)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> pipeline.max_len: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1]-1]
            records.append({'prompt':prompt, 'answer':label})

        pred_answers = batch_infer(pipeline, tokenizer, [record['prompt'] for record in records], cfg.batch_size)
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))


if __name__ == "__main__":
    main()