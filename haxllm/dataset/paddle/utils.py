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
    

def create_paddle_loader(data, batch_size, train):
    drop_last = train
    data = {**data, 'mask': np.ones(n, dtype=np.bool_)}
    n = next(iter(data.values())).shape[0]
    if drop_last:
        steps_per_epoch = n // batch_size
    else:
        steps_per_epoch = math.ceil(n // batch_size)
    dataset = TensorDictDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, drop_last=drop_last)
    return data_loader, steps_per_epoch
