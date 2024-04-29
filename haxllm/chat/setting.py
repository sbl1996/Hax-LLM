import inspect
from typing import Sequence, Optional, Dict, Tuple, Type


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
