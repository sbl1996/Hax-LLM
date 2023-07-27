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
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer

import jax
import jax.numpy as jnp
from haxllm.chat.inference import ChatIO, chat_loop
from haxllm.pipeline.text_generation import ChatPipeline


class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

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
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _(event):
        event.app.current_buffer.newline()

    def __init__(self, multiline: bool = False, mouse: bool = False):
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(
            words=["!!exit", "!!reset"], pattern=re.compile("$")
        )
        self._console = Console()
        self._multiline = multiline
        self._mouse = mouse

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            mouse_support=self._mouse,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.bindings if self._multiline else None,
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
        contents = ""
        # `end_sequence` signals the end of a message. It is unlikely to occur in
        #  message content.
        end_sequence = " __END_OF_A_MESSAGE_47582648__\n"
        len_end = len(end_sequence)
        while True:
            if len(contents) >= len_end:
                last_chars = contents[-len_end:]
                if last_chars == end_sequence:
                    break
            try:
                char = sys.stdin.read(1)
                contents = contents + char
            except EOFError:
                continue
        contents = contents[:-len_end]
        print(f"[!OP:{role}]: {contents}", flush=True)
        return contents

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

    model_name = getattr(cfg, "model", None)
    if model_name is None:
        raise ValueError("Model name not specified")

    template = cfg.template
    # TODO: infer template from model name
    # template = getattr(cfg, "template", None)
    # if template is None:
    #     print("Template not specified, infer from model name...")
    #     from hydra.core.hydra_config import HydraConfig
    #     hydra_cfg = HydraConfig.get()
    #     main_path = [ s['path'] for s in hydra_cfg.runtime['config_sources'] if s['provider'] == 'main' ][0]
    #     all_templates = [
    #         os.path.basename(p).replace(".yaml", "")
    #         for p in glob.glob(os.path.join(main_path, "template", "*.yaml"))
    #     ]
    #     template_names = difflib.get_close_matches(model_name, all_templates, n=1, cutoff=0.5)
    #     if len(template_names) == 0:
    #         raise ValueError(f"Cannot infer template from model name {model_name}")
    #     template = template_names[0]

    template_config = OmegaConf.to_container(template, resolve=True)
    random_seed = cfg.seed
    tokenizer_name = template_config.pop("tokenizer")
    conv_template = template_config.pop("conv_template", None)

    checkpoint = getattr(cfg, "checkpoint", None)
    if checkpoint is None:
        checkpoint = model_name + "_np.safetensors"
        print(f"Checkpoint not specified, follow model config, using {checkpoint}")

    temperature = getattr(cfg, "temperature", 1.0)
    top_p = getattr(cfg, "top_p", 1.0)
    top_k = getattr(cfg, "top_k", -1)
    debug = cfg.debug

    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    # support for both cpu, gpu and tpu
    
    platform = jax.default_backend()

    mesh = getattr(cfg, "mesh", None)
    if mesh == 'auto':
        mesh = None if platform == 'cpu' else [1, jax.local_device_count()]

    dtype = cfg.dtype
    if dtype == 'auto':
        dtype = jnp.bfloat16 if platform == 'tpu' else jnp.float16

    param_dtype = cfg.param_dtype
    if param_dtype == 'auto':
        param_dtype = jnp.bfloat16 if platform == 'tpu' else jnp.float16


    print(f"Loading tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.decode(tokenizer("init")['input_ids'])
    print("Load tokenizer {}".format(time.time() - start))

    parallel = mesh is not None

    module = "haxllm.model"
    peft = getattr(cfg, "peft", None)
    if peft is not None:
        module = module + "." + peft
    mod = importlib.import_module(module + "." + template_config.pop("family"))

    model_config = {"name": model_name, **template_config}
    config = getattr(mod, "load_config")(
        dtype=jnp.dtype(dtype),
        param_dtype=jnp.dtype(param_dtype),
        **model_config,
        decode=True,
        shard=parallel,
        shard_cache=parallel,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id

    model = getattr(mod, "TransformerLMHeadModel")(config)

    max_len = getattr(cfg, "max_len", None) or config.n_positions
    pipeline = ChatPipeline(
        tokenizer, model, max_len=max_len, seed=random_seed)
    pipeline.init(transformer_weight=checkpoint, mesh=mesh)

    print("Conversation setting:")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  top_k: {top_k}")
    print(f"  max_len: {max_len}")
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