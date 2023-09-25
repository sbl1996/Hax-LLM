import importlib


def get_module(family, peft=None):
    if peft:
        if peft == 'ptuning':
            mod = importlib.import_module("haxllm.model.ptuning." + family)
        elif peft == 'lora':
            mod = importlib.import_module("haxllm.model.lora." + family)
        elif peft == 'llama_adapter':
            mod = importlib.import_module("haxllm.model.llama_adapter." + family)
        else:
            raise ValueError(f"Unknown PEFT method: {peft}")
    else:
        mod = importlib.import_module("haxllm.model." + family)
    return mod
