# Mask 

## Training

### padding_left
not allowed (because no need)

### padding_right
padding_mask
- not used in model

causal mask
- generated and used in attention

sliding window mask
- generated and used in attention


## Decode

### padding_left (batch)
padding_mask
- prefill: generate at model start, used in attention's decode_for_padding to generate cached positions
- decode: not generated, rely on cached positions to generate padding mask

causal mask
- generated and used in attention


sliding window mask
- generated and used in attention

### padding_right (single)
padding_mask
- not used in model

causal mask
- generated and used in attention

sliding window mask
- generated and used in attention
