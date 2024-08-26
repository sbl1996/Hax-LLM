# Hax-LLM
--------------------------------------------------------------------------------

`Hax-LLM` is Hastur's experiments in scaling LLM to 10B+ parameters with JAX and TPUs.


## Models
- LLaMA (1, 2, 3, 3.1)
- Mistral (v0.1, v0.3)
- ChatGLM (2, 3)
- Yi (v1, v1.5)
- InternLM (v1, v2.5)
- Qwen (v1, v1.5, v2)
- Phi (3.5)


## Parameter-Efficient Fine-Tuning (PEFT)
- LoRA
- P-tuning v2
- LLaMA-Adapter


## Supported Tasks
- Sequence classification
- Language modeling
- Visual question answering (in process)


## Training Features
- Data and model parallel 
- Mixed precision
- Gradient checkpoint
- Resource monitoring


## Inference Features
- Model parallel
- Beam search
- Temperature, top-k, top-p
- KV cache
- Quantization


## Experiments
Check the experiments and training scripts on this [repo](https://github.com/sbl1996/llm_experiments).

## Convert Checkpoints

We should download the model and convert the checkpoints to JAX format.
```bash
python -m haxllm.model.dump --source mistralai/Mistral-7B-Instruct-v0.3
```
`mistralai/Mistral-7B-Instruct-v0.3` is the model name, can be huggingface model name, local directory or checkpoint file (pytorch-model-*.bin or model.safetensors).

```bash
rm -rf ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3
```
You can remove the cache to save disk space.


## Mock OpenAI API

Then, we can start a server with the converted model to mock OpenAI API.
```bash
python -m haxllm.chat.openai_api template=mistral model=mistral-7b-v0.3 checkpoint=chatglm2-6b_np.safetensors \
    max_len=4096 temperature=0.8 top_p=0.9 max_new_tokens=1000
```

Then, we can chat with the model via OpenAI API (streaming supported).
```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "none"

req = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Hello"}
    ],
    stream=False
)
print(req.choices[0].message.content)


for chunk in openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True
):
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Sponsors
This work is supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).
