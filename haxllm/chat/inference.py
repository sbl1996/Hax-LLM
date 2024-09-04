import numpy as np
import jax.numpy as jnp

from haxllm.pipeline.text_generation import ChatPipeline
from haxllm.model.decode import sample_token, split_rng


def partial_stop(output, stop_str):
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


def generate_stream(pipeline: ChatPipeline, params, stream_interval=2, kv_cache=None):
    tokenizer = pipeline.tokenizer
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature") or 1.0)
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    max_len = int(params.get("max_len") or 2048)
    top_p = float(params.get("top_p") or 1.0)
    top_k = int(params.get("top_k") or -1)  # -1 means disable
    echo = bool(params.get("echo") or False)
    stop_token_ids = list(params.get("stop_token_ids", [])) or pipeline.stop_token_ids
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)
    live_seq = np.full((max_len,), fill_value=tokenizer.pad_token_id, dtype=np.int32)
    offset = len(input_ids)
    live_seq[:offset] = input_ids

    max_allowed = max_len - input_echo_len - 8
    max_new_tokens = int(params.get("max_new_tokens") or max_allowed)
    max_new_tokens = min(max_new_tokens, max_allowed)

    # TODO: why 8?
    max_src_len = max_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    pre = pipeline.get_cache_index()

    if temperature < 1e-5 or top_k == 1:
        rng = None
    else:
        rng = pipeline.get_next_rng()

    for i in range(max_new_tokens):
        if i == 0:
            input_ids = jnp.array([input_ids[pre:]], dtype=jnp.int32)
            logits = pipeline.stream_forward(input_ids)
        else:
            logits = pipeline.stream_forward(
                jnp.array([[token]], dtype=jnp.int32))

        rng, subrng = split_rng(rng)
        token = sample_token(logits[0, -1, :], live_seq, subrng, temperature, top_p, top_k, repetition_penalty) 
        token = int(token)

        output_ids.append(token)
        live_seq[offset] = token
        offset += 1
        # print(f"[{i}] token {token} {tokenizer.convert_ids_to_tokens([token])}")

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            partially_stopped = False
            if not partially_stopped:
                usage = {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                }
                yield {
                    "text": output,
                    "usage": usage,
                    "finish_reason": None,
                }

        if stopped:
            break


    # finish stream event, which contains finish reason

    # TODO: local variable 'i' referenced before assignment
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }
