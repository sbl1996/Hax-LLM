import math
import numpy as np


def create_tfds(data, batch_size, train=True, seed=None):
    import tensorflow as tf
    drop_remainder = train
    n = next(iter(data.values())).shape[0]
    data = {**data, "mask": np.ones(n, dtype=np.bool_)}
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.cache()
    if train:
        ds = ds.shuffle(1000000, seed=seed)
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=train)
    if drop_remainder:
        steps_per_epoch = n // batch_size
    else:
        steps_per_epoch = math.ceil(n // batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds, steps_per_epoch