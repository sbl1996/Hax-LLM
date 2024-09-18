from typing import Iterable, Sequence, Literal
import random

import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tokenizers import Tokenizer

import numpy as np
import jax

from haxllm.utils import pad, time_now


def evaluate_chinese(
    predict_fn,
    tokenizer: Tokenizer,
    ds_eval: Iterable,
    eval_steps: int,
    batch_size: int,
    stop_token_ids: Sequence[int],
    echo: Literal[False, 'first', 'all', 'random'] = False,
    print_metrics: Literal[False, 'step', 'final'] = False,
):
    assert echo in [False, 'first', 'all', 'random']
    assert print_metrics in ['step', 'final', False]
    eval_iter = iter(ds_eval)
    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

    if echo == 'random':
        echo_step = random.randint(0, eval_steps - 1)
    for i in range(eval_steps):
        eval_batch = next(eval_iter)
        eval_batch = jax.tree_util.tree_map(
            lambda x: pad(x._numpy(), batch_size), eval_batch)

        input_ids = eval_batch["input_ids"]
        labels = eval_batch["labels"]

        output_ids = predict_fn(input_ids)

        input_lens = np.argmax(input_ids == tokenizer.pad_token_id, axis=1)
        input_lens = np.where(input_lens == 0, input_ids.shape[1], input_lens)
        output_ids = [
            output_ids[i, input_lens[i]:] for i in range(output_ids.shape[0])
        ]

        decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        stop_tokens = tokenizer.convert_ids_to_tokens(stop_token_ids)
        for stop_token in stop_tokens:
            if stop_token == '':
                continue
            decoded_preds = [pred.split(stop_token)[0] for pred in decoded_preds]

        if i == eval_steps - 1:
            n = eval_batch['mask'].sum()
            decoded_inputs = decoded_inputs[:n]
            decoded_preds = decoded_preds[:n]
            decoded_labels = decoded_labels[:n]

        for k, (input, pred, label) in enumerate(zip(decoded_inputs, decoded_preds, decoded_labels)):
            if echo == 'all' or (echo == 'first' and i == 0 and k == 0) or (echo == 'random' and i == echo_step and k == 0):
                print("=====================================")
                print("input:")
                print(input)
                print("pred:")
                print(pred)
                print("label:")
                print(label)
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            try:
                scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
                result = scores[0]
                
                for k, v in result.items():
                    score_dict[k].append(round(v["f"] * 100, 4))

                bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
                score_dict["bleu-4"].append(round(bleu_score * 100, 4))
            except ValueError:
                print("=====================================")
                print("input:")
                print(tokenizer.decode(input_ids[k], skip_special_tokens=False))
                print("pred:")
                print(pred)
                print("label:")
                print(label)
                print("hypothesis:")
                print(hypothesis)
                print("reference:")
                print(reference)
                for k in score_dict.keys():
                    score_dict[k].append(0)
        if print_metrics == 'step':
            print(f'\r{i+1}/{eval_steps} {", ".join([ f"{k}: {np.mean(v):.4f}" for k, v in score_dict.items() ])}')
    if print_metrics == 'final':
        print(f'{"-"*20} Final {"-"*20}')
        print(f'{time_now()} {i+1}/{eval_steps} {", ".join([ f"{k}: {np.mean(v):.4f}" for k, v in score_dict.items() ])}')
    return score_dict
