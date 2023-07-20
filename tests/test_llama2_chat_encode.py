from transformers import AutoTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# from https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L224
def encode1(tokenizer, dialog):
    if dialog[0]["role"] != "system":
        dialog = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + dialog
    dialog = [
        {
            "role": dialog[1]["role"],
            "content": B_SYS
            + dialog[0]["content"]
            + E_SYS
            + dialog[1]["content"],
        }
    ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    BOS = tokenizer.bos_token
    EOS = tokenizer.eos_token
    dialog_tokens = sum(
        [
            tokenizer.encode(
                f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}",
                add_special_tokens=False,
            )
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ],
        [],
    )
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    dialog_tokens += tokenizer.encode(
        f"{BOS}{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        add_special_tokens=False,
    )
    return dialog_tokens


def encode2(tokenizer, dialog):
    BOS = tokenizer.bos_token
    EOS = tokenizer.eos_token
    if dialog[0]["role"] != "system":
        dialog = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + dialog
    dialog = [
        {
            "role": dialog[1]["role"],
            "content": B_SYS
            + dialog[0]["content"]
            + E_SYS
            + dialog[1]["content"],
        }
    ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    ret = "".join(
        [
            f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ]
    )
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    ret += f"{BOS}{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
    return tokenizer.encode(ret, add_special_tokens=False)


access_token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=access_token)

dialogs = [
    [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    [
        {"role": "user", "content": "AA?"},
        {"role": "assistant", "content": "BB"},
        {"role": "user", "content": "CC?"},
        {"role": "assistant", "content": "DD"},
        {"role": "user", "content": "EE?"},
    ],
    [
        {"role": "system", "content": "Always answer with Haiku"},
        {"role": "user", "content": "I am going to Paris, what should I see?"},
    ],    
]

tokens1 = [encode1(tokenizer, dialog) for dialog in dialogs]
tokens2 = [encode2(tokenizer, dialog) for dialog in dialogs]


for i in range(len(dialogs)):
    assert tokens1[i] == tokens2[i], f"tokens1[{i}] != tokens2[{i}], dialog: {dialogs[i]}"