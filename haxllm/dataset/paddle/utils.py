import numbers
from typing import Mapping, Sequence
import math
import numpy as np
from paddle.io import Dataset, DataLoader


class TensorDictDataset(Dataset):

    def __init__(self, tensor_dict):
        self.tensor_dict = tensor_dict

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensor_dict.items()}

    def __len__(self):
        return next(iter(self.tensor_dict.values())).shape[0]
    

def default_collate_fn(batch):
    sample = batch[0]
    if isinstance(sample, (np.ndarray, np.bool_)):
        batch = np.stack(batch, axis=0)
        return batch
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch
    elif isinstance(sample, (str, bytes)):
        return batch
    elif isinstance(sample, Mapping):
        return {
            key: default_collate_fn([d[key] for d in batch])
            for key in sample
        }
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError(
                "fileds number not same among samples in a batch")
        return [default_collate_fn(fields) for fields in zip(*batch)]

    raise TypeError("batch data con only contains: tensor, numpy.ndarray, "
                    "dict, list, number, but got {}".format(type(sample)))


def create_paddle_loader(data, batch_size, train, num_workers):
    drop_last = train
    n = next(iter(data.values())).shape[0]
    data = {**data, 'mask': np.ones(n, dtype=np.bool_)}
    if drop_last:
        steps_per_epoch = n // batch_size
    else:
        steps_per_epoch = math.ceil(n // batch_size)
    dataset = TensorDictDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, drop_last=drop_last, collate_fn=default_collate_fn)
    return data_loader, steps_per_epoch
