# Hax-LLM
--------------------------------------------------------------------------------

`Hax-LLM` is Hastur's experiments in scaling LLM to 10B+ parameters with JAX and TPUs.


## Models
- GPT-2
- BERT (RoBERTa)
- LLaMA (1, 2)
- Vicuna
- ChatGLM2
- InternLM
- Qwen (in process)


## Parameter-Efficient Fine-Tuning (PEFT)
- LoRA
- P-tuning v2
- LLaMA-Adapter (in process)


## Supported Tasks
- Sequence classification
- Language modeling
- Visual question answering (in process)


## Training Features
- Data and model parallel 
- Mixed precision
- Gradient checkpoint (fine-grained)
- Scan (for faster jit compilation)
- Model parameter freezing
- Memory-efficient attention
- Resource monitoring


## Inference Features
- Model parallel
- Beam search
- Temperature, top-k, top-p
- KV cache

## Experiments
Check the experiments and training scripts on this [repo](https://github.com/sbl1996/llm_experiments).

## Chat CLI

### Vicuna

First, we should download the model and convert the checkpoints to JAX format.
```bash
python -m haxllm.model.dump -m llama -s lmsys/vicuna-7b-v1.3
```
`llama` is the model family, currently supports `gpt2`, `llama` and `chatglm2`.
`lmsys/vicuna-7b-v1.3` is the model name, can be huggingface model name, local directory or checkpoint file (pytorch-model-*.bin or model.safetensors).

Then, we can chat with the model.
```bash
python -m haxllm.chat.cli template=llama model=vicuna-7b checkpoint=vicuna-7b-v1.3_np.safetensors temperature=0.7
```
You may refer `configs/chat/base.yaml` for more settings like max length, temperature, top-k, top-p.

```bash
rm -rf ~/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.3
```
You can remove the cache to save disk space.

### LLaMA-2

First, we should download the model and convert the checkpoints to JAX format.
```bash
python -m haxllm.model.dump -m llama -s meta-llama/Llama-2-7b-chat-hf -t safetensors
```
LLaMA-2 has both `safetensors` and `bin` checkpoints.  We want to download the `safetensors` only to save disk space.

Good to go!
```bash
python -m haxllm.chat.cli template=llama2 model=llama2-7b checkpoint=Llama-2-7b-chat-hf_np.safetensors
```

## Mock OpenAI API

We can also start a server to mock OpenAI API.
```bash
python -m haxllm.chat.openai_api template=chatglm2 model=chatglm2-6b checkpoint=chatglm2-6b_np.safetensors
```

Then, we can chat with the model via OpenAI API (streaming supported).
```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "none"

req = openai.ChatCompletion.create(
    model="chatglm2-6b",
    messages=[
        {"role": "user", "content": "Hello"}
    ],
    stream=False
)
print(req.choices[0].message.content)


for chunk in openai.ChatCompletion.create(
    model="chatglm2-6b",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True
):
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## LangChain

With the OpenAI API mocking server, `langchain` support is enabled.
```python
from langchain.chat_models import ChatOpenAI

openai_api_base = "http://localhost:8000/v1"
openai_api_key = "none"

chat_model = ChatOpenAI(
    openai_api_base=openai_api_base, openai_api_key=openai_api_key)
print(chat_model.predict("Hello"))
```

## Sponsors
This work is supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).
