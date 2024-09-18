import inspect
from typing import Sequence, Optional, Dict, Tuple, Type

from datetime import datetime


class ChatSetting:
    name: str
    roles: Tuple[str, str]
    system: Optional[str] = None
    stop_token_ids: Optional[Sequence[int]] = None

    def get_prompt(self, messages: Sequence[Sequence[str]]):
        raise NotImplementedError
    
    def build_prompt(self, query):
        messages = [
            [self.roles[0], query],
            [self.roles[1], None],
        ]
        return self.get_prompt(messages)

    def find_prompt_prefix_suffix(self, tokenizer):
        if not hasattr(self, "_prefix_suffix_cache"):
            self._prefix_suffix_cache = {}
        cache = self._prefix_suffix_cache
        if tokenizer.name in cache:
            return self._prefix_suffix_cache[tokenizer.name]
        examples = ["你好", "晚上吃什么", "请问你是谁", "How are you?"]
        examples = [self.build_prompt(e) for e in examples]
        input_ids = [tokenizer.encode(e, add_special_tokens=True) for e in examples]
        l = 0
        while True:
            if len(set(input[l] for input in input_ids)) != 1:
                break
            l += 1
        r = 0
        while True:
            if len(set(input[-r-1] for input in input_ids)) != 1:
                break
            r += 1
        prefix, suffix = input_ids[0][:l], input_ids[0][-r:]
        cache[tokenizer.name] = prefix, suffix
        return prefix, suffix

    @classmethod
    def none(cls):
        return NoneChatSetting()


chat_settings: Dict[str, ChatSetting] = {}


def get_chat_setting(name: str) -> ChatSetting:
    name = name.lower()
    return chat_settings[name]()


def _register_chat_setting(
    setting: Type, force: bool = False) -> None:
    if not inspect.isclass(setting):
        raise TypeError('module must be a class, '
                        f'but got {type(setting)}')
    setting_name = setting.name.lower()
    if not force:
        assert setting_name not in chat_settings, f"{setting_name} has been registered."

    chat_settings[setting_name] = setting


def register_chat_setting(force: bool = False):
    def _register(setting):
        mixed_class = type(setting.__name__, (setting, ChatSetting), {})
        _register_chat_setting(setting=mixed_class, force=force)
        return mixed_class
    return _register


@register_chat_setting()
class NoneChatSetting:
    name = "non_chat"
    system = ""
    roles = ("USER", "ASSISTANT")
    stop_token_ids = ()

    def get_prompt(self, messages):
        if self.system:
            ret = self.system + '\n'
        else:
            ret = ""
        if len(messages) > 2:
            print("Warning: more than 2 messages in NON_CHAT mode.")
        for role, message in messages:
            if message:
                ret += message
        return ret


@register_chat_setting()
class VicunaChatSetting:
    name = "vicuna_v1"
    system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    roles = ("USER", "ASSISTANT")
    stop_token_ids = (2,)

    def get_prompt(self, messages):
        seps = [" ", "</s>"]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret


@register_chat_setting()
class YiChatSetting:
    name = "yi-chat"
    system = ""
    roles = ("user", "assistant")
    stop_token_ids = (2, 7,)

    def get_prompt(self, messages):
        bos = "<|im_start|>"
        eos = "<|im_end|>"
        ret = ""
        for i, (role, message) in enumerate(messages):
            ret += f"{bos}{role}\n"
            if message:
                ret += f"{message}{eos}\n"
            else:
                assert i == len(messages) - 1 and role == self.roles[1]
        return ret


@register_chat_setting()
class LLaMA3ChatSetting:
    name = "llama3-instruct"
    system = ""
    roles = ("user", "assistant")
    stop_token_ids = (128001, 128009,)
    add_date = False

    def get_prompt(self, messages):
        boh = "<|start_header_id|>"
        eoh = "<|end_header_id|>"
        eos = "<|eot_id|>"
        # ret = "<|begin_of_text|>"
        ret = ""
        system = self.system
        if messages[0][0] == "system":
            system = messages[0][1]
            messages = messages[1:]
        system = system.strip()
        date_prompt = ""
        if self.add_date:
            date_string = datetime.now().strftime('%d %b %Y')
            # date_string = "26 Jul 2024"
            date_prompt = \
                f"Cutting Knowledge Date: December 2023\n" \
                f"Today Date: {date_string}\n\n"
            system = date_prompt + system
        if system:
            ret += f"{boh}system{eoh}\n\n{system}{eos}"
        for i, (role, message) in enumerate(messages):
            ret += f"{boh}{role}{eoh}\n\n"
            if message:
                ret += message.strip() + eos
            else:
                assert i == len(messages) - 1 and role == self.roles[1]
        return ret


@register_chat_setting()
class LLaMA31InstructSetting(LLaMA3ChatSetting):
    name = "llama3.1-instruct"
    add_date = True


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

@register_chat_setting()
class LLaMA2ChatSetting:
    name = "llama2-chat"
    system = DEFAULT_SYSTEM_PROMPT
    roles = ("user", "assistant")
    stop_token_ids = (2,)

    def get_prompt(self, messages):
        # TODO: add support for custom system prompt
        BOS = "<s>"
        EOS = "</s>"
        dialog = [
            {"role": role, "content": message}
            for role, message in messages
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


@register_chat_setting()
class MistralChatSetting:
    name = "mistral-instruct"
    system = ""
    roles = ("user", "assistant")
    stop_token_ids = (2,)
    extra_space = True

    def get_prompt(self, messages):
        B_INST, E_INST = "[INST]", "[/INST]"
        eos = "</s>"
        # ret = "<s>"
        ret = ""
        system = self.system
        if messages[0][0] == "system":
            system = messages[0][1]
            messages = messages[1:]
        system = system.strip()
        n = len(messages)
        last_msg = messages[-1]
        if (last_msg[0] == self.roles[1] and last_msg[1] is None) or last_msg[0] == self.roles[0]:
            system = system.strip()
        else:
            system = None
        space = " " if self.extra_space else ""
        for i, (role, content) in enumerate(messages):
            if role == self.roles[0]:
                if i % 2 != 0:
                    raise ValueError("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...")
                if system and i in [n - 1, n - 2]:
                    ret += f"{B_INST}{space}{system}\n\n{content}{E_INST}"
                else:
                    ret += f"{B_INST}{space}{content}{E_INST}"
            else:
                if content:
                    ret += f"{space}{content}{eos}"
                else:
                    ret += space
        return ret


@register_chat_setting()
class MistralNemoChatSetting(MistralChatSetting):
    name = "mistral-nemo"
    extra_space = False


@register_chat_setting()
class InternLMChatSetting:
    name = "internlm"
    system = ""
    roles = ("<|User|>", "<|Bot|>")
    stop_token_ids = (2, 103028)

    def get_prompt(self, messages):
        r'''
        prompt = ""
        for record in history:
            prompt += f"""<s><|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n"""
        if len(prompt) == 0:
            prompt += "<s>"
        prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""            
        '''
        sep = "<eoh>\n"
        sep2 = "<eoa>\n"
        ret = ""
        m = messages
        n = len(m)
        n = n - 2 if n % 2 == 0 else n - 1
        for i in range(0, n, 2):
            role1, message1 = m[i]
            role2, message2 = m[i+1]
            ret += f"""<s>{role1}:{message1}{sep}{role2}:{message2}{sep2}"""
        if len(ret) == 0:
            assert len(m) - n == 2
            role1, message1 = m[-2]
            role2, message2 = m[-1]
            ret += f"""<s>{role1}:{message1}{sep}{role2}:"""
            if message2:
                ret += message2 + sep2
        else:
            if len(m) - n == 1:
                role, message = m[-1]
                ret += f"""<s>{role}:{message}{sep2}"""
            else:
                role1, message1 = m[-2]
                role2, message2 = m[-1]
                ret += f"""<s>{role1}:{message1}{sep}{role2}:"""
                if message2:
                    ret += message2 + sep2
        return ret


@register_chat_setting()
class InternLM2ChatSetting:
    name = "internlm2"
    system = "You are an AI assistant whose name is InternLM (书生·浦语).\n" \
    "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory " \
    "(上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n" \
    "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such " \
    "as English and 中文."
    roles = ("user", "assistant")
    stop_token_ids = (2, 92542)

    def get_prompt(self, messages):
        bot, eot = "<|im_start|>", "<|im_end|>"
        # ret = "<s>"
        ret = ""
        system = self.system
        if messages[0][0] == "system":
            system = messages[0][1]
            messages = messages[1:]
        system = system.strip()
        if system:
            ret += f"{bot}system\n{system}{eot}\n"
        for i, (role, message) in enumerate(messages):
            ret += f"{bot}{role}\n"
            if message:
                ret += f"{message}{eot}\n"
            else:
                assert i == len(messages) - 1 and role == self.roles[1]
        return ret


@register_chat_setting()
class ChatGLM2ChatSetting:
    name = "chatglm2"
    system = ""
    roles = ("问", "答")
    stop_token_ids = (0, 2)

    def get_prompt(self, messages):
        sep = "\n\n"
        if self.system:
            ret = self.system + sep
        else:
            ret = ""
        for i, (role, message) in enumerate(messages):
            if i % 2 == 0:
                round = i // 2 + 1  # only difference from CHAT_GLM
                ret += f"[Round {round}]{sep}{role}：{message}"
            else:
                ret += f"{sep}{role}："
                if message:
                    ret += message + sep
        return ret


def qwen_encode_message(self, messages):
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    system = self.system
    if messages[0][0] == "system":
        system = messages[0][1]
        messages = messages[1:]
    system = system.strip()

    sep = "\n"
    ret = ""
    if system:
        ret += f"{im_start}system{sep}{system}{im_end}"

    for i, (role, message) in enumerate(messages):
        if i % 2 == 0:
            ret += f"{im_start}{role}{sep}{message}{im_end}{sep}"
        else:
            ret += f"{im_start}{role}{sep}"
            if message:
                ret += f"{message}{im_end}{sep}"
            else:
                assert i == len(messages) - 1 and role == self.roles[1]
    return ret


@register_chat_setting()
class QwenChatSetting:
    name = "qwen"
    system = "You are a helpful assistant."
    roles = ("user", "assistant")
    stop_token_ids = (151643,)

    def get_prompt(self, messages):
        return qwen_encode_message(self, messages)


@register_chat_setting()
class Qwen2ChatSetting:
    name = "qwen2"
    system = "You are a helpful assistant"
    roles = ("user", "assistant")
    stop_token_ids = (151643,)

    def get_prompt(self, messages):
        return qwen_encode_message(self, messages)


@register_chat_setting()
class Phi3ChatSetting:
    name = "phi3"
    system = ""
    roles = ("<|user|>", "<|assistant|>")
    stop_token_ids = (32000, 32001, 32007)

    def get_prompt(self, messages):
        eot = "<|end|>"
        system = self.system
        if messages[0][0] == "system":
            system = messages[0][1]
            messages = messages[1:]
        system = system.strip()
        ret = ""
        if system:
            ret += f"<|system|>\n{system}{eot}\n"
        for i, (role, message) in enumerate(messages):
            ret += f"{role}\n"
            if message:
                ret += f"{message}{eot}\n"
            else:
                assert i == len(messages) - 1 and role == self.roles[1]
        return ret


@register_chat_setting()
class GLM4ChatSetting:
    name = "glm4"
    system = ""
    roles = ("<|user|>", "<|assistant|>")
    stop_token_ids = (151329, 151336, 151338)

    def get_prompt(self, messages):
        system = self.system
        if messages[0][0] == "system":
            system = messages[0][1]
            messages = messages[1:]
        system = system.strip()
        # ret = "[gMASK]<sop>"
        ret = ""
        if system:
            ret += f"<|system|>\n{system}"
        for i, (role, message) in enumerate(messages):
            ret += f"{role}\n"
            if message:
                ret += message
            else:
                assert i == len(messages) - 1 and role == self.roles[1]
        return ret