from transformers import AutoTokenizer
import haxllm.model.llama
from haxllm.chat.conversation import get_conv_template

model_id = "mistralai/Mistral-Nemo-Instruct-2407"
conv_template = "mistral-nemo"
print(f"{model_id} <-> {conv_template}")

t = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "Please tell me the current weather in New York City."},
]

n = len(msgs)
for start, end in [(0, n), (0, -1), (1, n)]:
    t_msgs = msgs[start:end]
    input_ids1 = t.apply_chat_template(
        t_msgs, tokenize=True, add_generation_prompt=False)
    cs = get_conv_template(conv_template).config
    ms = [(m['role'], m['content']) for m in t_msgs]
    input_ids2 = t(cs.get_prompt(ms))['input_ids']
    res = input_ids1 == input_ids2
    if not res:
        print(t_msgs)
        print("chat_template:")
        print(t.apply_chat_template(t_msgs, tokenize=False, add_generation_prompt=False))
        print(input_ids1)
        print("\n\n\nchat setting:")
        print(cs.get_prompt(ms))
        print(input_ids2)