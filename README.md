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

## Sponsors
This work is supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

## Logo
The logo of `Hax-LLM` is designed by Adobe Firefly. Amazing!