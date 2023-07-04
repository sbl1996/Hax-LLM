![Hax-LLM Logo](/docs/_static/hax-llm-2.jpg)

--------------------------------------------------------------------------------

`Hax-LLM` is Hastur's experiments in scaling LLM to 10B+ parameters with JAX and TPUs.


## Models
- GPT-2
- BERT (RoBERTa)
- LLaMA (Vicuna)
- ChatGLM (v2 in process)


## Parameter-Efficient Fine-Tuning (PEFT)
- LoRA
- P-tuning v2
- LLaMA-Adapter (in process)


## Supported Tasks
- Sequence classification
- Language modeling


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

## Chat

### Vicuna

First, we should download the model and convert the checkpoints to JAX format.
```bash
python3 -m haxllm.model.dump -m llama -s lmsys/vicuna-7b-v1.3
```
`llama` is the model family, currently supports `gpt2`, `llama` and `chatglm2`.
`lmsys/vicuna-7b-v1.3` is the model name, can be huggingface model name, local directory or checkpoint file (pytorch-model-*.bin or model.safetensors).

Then, we can chat with the model.
```bash
python3 -m haxllm.chat.cli model=vicuna-7b checkpoint=vicuna-7b-v1.3_np.safetensors
```
You may refer `configs/chat/base.yaml` for more settings like max length, temperature, top-k, top-p.

```bash
rm -rf ~/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.3
```
You can remove the cache to save disk space.


## Sponsors
This work is supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

## Logo
The logo of `Hax-LLM` is designed by Adobe Firefly. Amazing!