from haxllm.dataset.imdb import create_dataset as create_imdb_dataset
from haxllm.dataset.sst2 import create_dataset as create_sst2_dataset


def create_dataset(name, tokenizer, max_len=128, eval_size=None, batch_size=128, eval_batch_size=None,
                   seed=42, with_test=False):
    if name == 'imdb':
        return create_imdb_dataset(tokenizer, max_len, eval_size, batch_size, eval_batch_size, seed, with_test)
    elif name == 'sst2':
        return create_sst2_dataset(tokenizer, max_len, eval_size, batch_size, eval_batch_size, seed, with_test)
    else:
        raise ValueError(f'Unknown dataset: {name}')
