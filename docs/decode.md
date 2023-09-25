# Decode mechanism

## Two stage
Two-stage decoding means that the decoding process is divided into two stages: the first stage is to decode the context (seq_len>1) and the second stage is to decode the target (seq_len=1). JAX jit will recompile every time the input shape changes, so the two-stage decoding without special treatment will cause overhead.

## Pad context
To avoid the overhead of jit recompilation in two-stage decoding, we pad the context in the right to a length (a multiple of `pad_multiple`) and use the mask to mask the padding part.

## Support

| padding_side | batch_size | two_stage | pad_context | comment                             |
| right        | 1          | False     | /           | Slow inference, no recompilation    |
| right        | 1          | True      | False       | Fast inference, with recompilation  |
| right        | 1          | True      | True        | Fast inference, few recompilation   |
| right        | n          | False     | /           | Slow inference, no recompilation    |
| right        | n          | True      | /           | Not supported                       |

| left         | 1          | /         | /           | Not supported, no need to pad       |
| left         | n          | False     | /           | Not supported                       |
| left         | n          | True      | False       | Fast inference, with recompilation  |
| left         | n          | True      | True        | Not supported                       |

## Use case

### Chat or completion
For chat or completion, the batch size is usually 1, right/no padding and two-stage decoding are recommended, while setting `pad_context` to `True` and `pad_multiple` to a proper value (e.g. 64, 128) can avoid recompilation and improve performance.

### Evaluation
For evaluation, the batch size is usually larger than 1, left pad the batch to a fixed length (e.g. 256) and default to two-stage decoding.
