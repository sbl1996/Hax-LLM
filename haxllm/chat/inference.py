import abc
from typing import Optional
import time

import jax.numpy as jnp

from haxllm.chat.conversation import get_conv_template
from haxllm.pipeline.text_generation import ChatPipeline
from haxllm.model.decode import sample_token, split_rng


def partial_stop(output, stop_str):
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


def generate_stream(pipeline: ChatPipeline, params, max_len=2048, stream_interval=2):
    tokenizer = pipeline.tokenizer
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature") or 1.0)
    # repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p") or 1.0)
    top_k = int(params.get("top_k") or -1)  # -1 means disable
    stop_str = params.get("stop")
    echo = bool(params.get("echo") or False)
    stop_token_ids = list(params.get("stop_token_ids")) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)

    max_new_tokens = int(params.get("max_new_tokens") or (max_len - input_echo_len - 8))

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
        token = sample_token(logits[0, -1, :], subrng, temperature, top_p, top_k) 
        token = int(token)

        output_ids.append(token)
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


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""


def chat_loop(
    pipeline: ChatPipeline,
    chatio: ChatIO,
    conv_template: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    debug: bool = False,
):
    def new_chat():
        if conv_template is None:
            print("No conversation template provided. Using non_chat, which is a single-turn conversation. "
                  "Use !!reset to reset history after each turn.")
            return get_conv_template("non_chat")
        return get_conv_template(conv_template)

    conv = new_chat()
    pipeline.reset_chat_state()

    error_code = "__END_OF_A_MESSAGE_47582648__"

    while True:
        try:
            inp = chatio.prompt_for_input(conv.config.roles[0])
        except EOFError:
            inp = error_code
        
        if inp == "":
            continue

        if inp == "!!exit" or inp == error_code:
            print("exit...")
            break

        if inp == "!!reset":
            print("resetting...")
            conv = new_chat()
            pipeline.reset_chat_state()
            continue
        
        conv.append_message(conv.config.roles[0], inp)
        conv.append_message(conv.config.roles[1], None)

        prompt = conv.get_prompt()

        gen_params = {
            "prompt": prompt,
            "temperature": pipeline.temperature,
            "top_k": pipeline.top_k,
            "top_p": pipeline.top_p,
            # "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop_token_ids": conv.config.stop_token_ids,
            "echo": False,
            "max_len": pipeline.max_len,
        }

        chatio.prompt_for_output(conv.config.roles[1])
        output_stream = generate_stream(pipeline, gen_params, pipeline.max_len)
        t = time.time()
        outputs = chatio.stream_output(output_stream)
        duration = time.time() - t
        conv.update_last_message(outputs.strip())

        if debug:
            num_tokens = len(pipeline.tokenizer.encode(outputs))
            msg = {
                "conv_template": conv.config.name,
                "prompt": prompt,
                "outputs": outputs,
                "speed (token/s)": round(num_tokens / duration, 2),
            }
            print(f"\n{msg}\n")
