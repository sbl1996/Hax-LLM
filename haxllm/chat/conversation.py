import dataclasses
from typing import Sequence

from haxllm.chat.setting import ChatSetting, get_chat_setting


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    config: ChatSetting
    messages: Sequence[Sequence[str]]

    def get_prompt(self) -> str:
        return self.config.get_prompt(self.messages)

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    # def to_openai_api_messages(self):
    #     """Convert the conversation to OpenAI chat completion format."""
    #     ret = [{"role": "system", "content": self.system}]

    #     for i, (_, msg) in enumerate(self.messages[self.offset :]):
    #         if i % 2 == 0:
    #             ret.append({"role": "user", "content": msg})
    #         else:
    #             if msg is not None:
    #                 ret.append({"role": "assistant", "content": msg})
    #     return ret

    def copy(self):
        return Conversation(
            config=self.config,
            messages=[[x, y] for x, y in self.messages],
        )


def get_conv_template(name):
    """Get a conversation template by name."""
    return Conversation(
        config=get_chat_setting(name),
        messages=[],
    )


def apply_template(messages, conv_template):
    conv = get_conv_template(conv_template)
    query = messages[-1]['content']
    prev_messages = messages[:-1]

    if len(prev_messages) > 0 and prev_messages[0]['role'] == "system":
        system_message = prev_messages.pop(0)['content']
    else:
        system_message = None

    if system_message:
        conv.config.system = system_message

    for msg in prev_messages:
        if msg['role'] == "user":
            conv.append_message(conv.config.roles[0], msg.content)
        elif msg['role'] == "assistant":
            conv.append_message(conv.config.roles[1], msg.content)
        elif msg['role'] == "system":
            raise ValueError("system message must be the first message")
    conv.append_message(conv.config.roles[0], query)
    conv.append_message(conv.config.roles[1], None)
    prompt = conv.get_prompt()
    return prompt