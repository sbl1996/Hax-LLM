# Quantization

## Benchmark
Qwen2-72B-Instruct-AWQ, max_length 1024
| mode      | int8  | uint4 | 
|-----------|-------|-------|
| prefill   |   227 | ERROR |
| decode    |     6 | ERROR |
| memory(G) |  84.7 |       |


Qwen2-1.5B-Instruct-AWQ, max_length 1024
| mode      |  bf16 |  int8 | uint4 | 
|-----------|-------|-------|-------|
| prefill   |  2800 |  2600 |  2550 |
| decode    |   155 |   115 |   105 |
| memory(G) |   3.2 |   2.1 |   1.5 |

## Known Issues

### LLO_CHECK failure
uint4 weight might raise this, either gptq or awq
Workaround: use p_dtype int8