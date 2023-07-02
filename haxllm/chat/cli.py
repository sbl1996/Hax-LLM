# pyright: reportUnboundVariable=false

import time
import os
import re
import sys
import importlib

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer

import jax.numpy as jnp
from haxllm.chat.inference import ChatIO, generate_stream, chat_loop
from haxllm.chat.conversation import get_conv_template
from haxllm.pipeline.text_generation import TextGenerationPipeline


class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)


class RichChatIO(ChatIO):
    def __init__(self):
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(
            words=["!!exit", "!!reset"], pattern=re.compile("$")
        )
        self._console = Console()

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role}:")

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        #  above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                if not outputs:
                    continue
                text = outputs["text"]
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return text


class ProgrammaticChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        print(f"[!OP:{role}]: ", end="", flush=True)
        contents = ""
        # `end_sequence` is a randomly-generated, 16-digit number
        #  that signals the end of a message. It is unlikely to occur in
        #  message content.
        end_sequence = "9745805894023423"
        while True:
            if len(contents) >= 16:
                last_chars = contents[-16:]
                if last_chars == end_sequence:
                    break
            try:
                char = sys.stdin.read(1)
                contents = contents + char
            except EOFError:
                continue
        return contents[:-16]

    def prompt_for_output(self, role: str):
        print(f"[!OP:{role}]: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)


@hydra.main(version_base=None, config_path="../../configs/chat", config_name="base")
def chat_app(cfg: DictConfig) -> None:
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.initialize_cache(os.path.expanduser("~/jax_cache"))
    import logging
    logging.getLogger("jax").setLevel(logging.WARNING)

    if cfg.style == "simple":
        chatio = SimpleChatIO()
    elif cfg.style == "rich":
        chatio = RichChatIO()
    elif cfg.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {cfg.style}")

    start = time.time()
    # jax_smi.initialise_tracking()

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    random_seed = cfg.seed
    tokenizer_name = model_config.pop("tokenizer")
    conv_template = model_config.pop("conv_template", None)

    checkpoint = getattr(cfg, "checkpoint", None)
    if checkpoint is None:
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        model_config_name = hydra_cfg.runtime.choices.model
        checkpoint = model_config_name + "_np.safetensors"
        print(f"Checkpoint not specified, follow model config, using {checkpoint}")

    temperature = getattr(cfg, "temperature", 1.0)
    top_p = getattr(cfg, "top_p", 1.0)
    top_k = getattr(cfg, "top_k", -1)
    debug = cfg.debug

    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    print(f"Loading tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.decode(tokenizer("init")['input_ids'])
    print("Load tokenizer {}".format(time.time() - start))

    mesh = getattr(cfg, "mesh", None)
    parallel = mesh is not None
    mod = importlib.import_module("haxllm.model." + model_config.pop("family"))

    config = getattr(mod, "load_config")(
        dtype=jnp.dtype(cfg.dtype),
        param_dtype=jnp.dtype(cfg.param_dtype),
        **model_config,
        decode=True,
        shard=parallel,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id

    model = getattr(mod, "TransformerLMHeadModel")(config)

    print("Load config {}".format(time.time() - start))

    max_len = getattr(cfg, "max_len", config.n_positions)
    pipeline = TextGenerationPipeline(
        tokenizer, model, mesh=mesh, max_len=max_len, seed=random_seed)
    pipeline.init(transformer_weight=checkpoint)

    try:
        chat_loop(
            pipeline,
            chatio,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_len=max_len,
            conv_template=conv_template,
            debug=debug,
        )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    chat_app()