import os
import time 

import json
from glob import glob

import pandas as pd

import hydra
from omegaconf import DictConfig

from haxllm.chat.config_utils import load_config
from haxllm.inference.batch import batch_inference
from haxllm.gconfig import set_gconfig


choices = ["A", "B", "C", "D"]


def accuracy(pred_answers, gold_answers):
    acc = 0
    for pred, gold in zip(pred_answers, gold_answers):
        if pred == gold:
            acc += 1
    return acc, len(gold_answers)


def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    task_accuracies = []
    for task in run_results:
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        acc, n = accuracy(pred_answers, gold_answers)
        task_accuracy = acc / n
        task_accuracies.append(task_accuracy)
        print("ACC-%s: %.4f" % (task, task_accuracy))
        total_acc += acc
        total_num += n
    micro_avg_acc = total_acc / total_num
    macro_avg_acc = sum(task_accuracies) / len(task_accuracies)
    print("ACC-all-micro: %.4f" % micro_avg_acc)
    print("ACC-all-macro: %.4f" % macro_avg_acc)


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


def truncate_prompt(tokenizer, prompt, max_len):
    while len(tokenizer.tokenize(prompt, add_special_tokens=True)) > max_len:
        prompt_split = prompt.split("\n\n")
        prompt_split.pop(1)
        prompt = '\n\n'.join(prompt_split)
    return prompt


def load(cfg):
    pipeline, conv_template = load_config(cfg, chat=False)
    return pipeline


@hydra.main(version_base=None, config_path="../configs/chat", config_name="base")
def main(cfg: DictConfig) -> None:
    from jax_smi import initialise_tracking
    initialise_tracking()
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/jax_cache"))

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
    compute_metric(output_filename)

    suffix = "_dev.csv"

    tasks = sorted(list(map(lambda x: x.split("/")[-1][:-len(suffix)], glob(os.path.join(data_dir, "dev", "*" + suffix)))))

    print("Use left padding for batch inference")
    cfg.padding_side = "left"
    pipeline = load(cfg)
    max_prompt_len = pipeline.max_len - pipeline.max_new_tokens
    tokenizer = pipeline.tokenizer
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
            prompt = truncate_prompt(tokenizer, prompt, max_prompt_len)
            label = test_df.iloc[i, test_df.shape[1]-1]
            records.append({'prompt': prompt, 'answer': label})

        inputs = [record['prompt'] for record in records]
        pred_answers = batch_inference(
            pipeline, inputs, cfg.batch_size, progress_bar=True)
        pred_answers = [x.strip() for x in pred_answers]
        gold_answers = [record['answer'] for record in records]

        acc, n = accuracy(pred_answers, gold_answers)
        print("ACC-%s: %.4f" % (task, acc / n))

        run_results[task] = {'pred_answers': pred_answers, 'gold_answers': gold_answers}

    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))


if __name__ == "__main__":
    main()