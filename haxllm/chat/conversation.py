import dataclasses
from enum import auto, Enum
from typing import Sequence, Optional, Dict


class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    ADD_NEW_LINE_SINGLE = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    CHAT_GLM = auto()
    NON_CHAT = auto()
    LLAMA2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system: str
    # Two roles
    roles: Sequence[str]
    # All messages. Each item is (role, message).
    messages: Sequence[Sequence[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: Optional[str] = None
    # Stop criteria (the default one is EOS token)
    stop_str: Optional[str] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: Optional[Sequence[int]] = None
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None

    # def get_next_input(self) -> str:
    #     """Get the next input for the model."""
    #     if self.sep_style == SeparatorStyle.ADD_COLON_TWO:
    #         seps = [self.sep, self.sep2]
    #         if len(self.messages) == 2:
    #             ret = self.system + self.sep
    #             messages = self.messages
    #         elif len(self.messages) > 2:
    #             assert len(self.messages) % 2 == 0 and self.messages[-1][1] is None
    #             ret = seps[1]
    #             messages = self.messages[-2:]
    #         else:
    #             raise ValueError(f"Invalid number of messages: {len(self.messages)}")
    #         for i, (role, message) in enumerate(messages):
    #             if message:
    #                 ret += role + ": " + message + seps[i % 2]
    #             else:
    #                 ret += role + ":"
    #         return ret
    #     else:
    #         raise ValueError(f"Invalid style: {self.sep_style}")

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.CHAT_GLM:
            sep = self.sep
            if self.system:
                ret = self.system + sep
            else:
                ret = ""
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i // 2}]{sep}{role}：{message}"
                else:
                    ret += f"{sep}{role}："
                    if message:
                        ret += message + sep
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            # TODO: add support for custom system prompt
            BOS = self.bos_token
            EOS = self.eos_token
            dialog = [
                {"role": role, "content": message}
                for role, message in self.messages
            ]
            if len(dialog) % 2 == 0:
                assert dialog[-1]["content"] is None
                dialog = dialog[:-1]
                append = True
            else:
                append = False
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
            if append:
                ret += f"{BOS}{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
            return ret
        elif self.sep_style == SeparatorStyle.NON_CHAT:
            if self.system:
                ret = self.system + '\n'
            else:
                ret = ""
            if len(self.messages) > 2:
                print("Warning: more than 2 messages in NON_CHAT mode.")
            for role, message in self.messages:
                if message:
                    ret += message
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."
    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# Non-chat template for models like Llama
register_conv_template(
    Conversation(
        name="non_chat",
        system="",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NON_CHAT,
        sep=" ",
    )
)


# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)


# ChatGLM-6B template
register_conv_template(
    Conversation(
        name="chatglm-6b",
        system="",
        roles=("问", "答"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHAT_GLM,
        sep="\n",
        stop_token_ids=[0],
    )
)


# ChatGLM2-6B template
register_conv_template(
    Conversation(
        name="chatglm2-6b",
        system="",
        roles=("问", "答"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHAT_GLM,
        sep="\n\n",
        stop_token_ids=[0],
    )
)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# Llama-2 chat template
register_conv_template(
    Conversation(
        name="llama2-chat",
        system=DEFAULT_SYSTEM_PROMPT,
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        bos_token="<s>",
        eos_token="</s>",
    )
)