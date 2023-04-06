import math
import os
import re

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def preprocess_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text


def load_imdb_data(data_folder):
    texts = []
    labels = []

    for sentiment in ['pos', 'neg']:
        sentiment_folder = os.path.join(data_folder, sentiment)
        for filename in os.listdir(sentiment_folder):
            with open(os.path.join(sentiment_folder, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(preprocess_text(text))
                labels.append(1 if sentiment == 'pos' else 0)

    return texts, labels


def encode_reviews(tokenizer, reviews, max_length):
    tokenized_reviews = tokenizer.batch_encode_plus(
        reviews,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='np'
    )
    return tokenized_reviews


def create_ds(reviews, batch_size, train=True, seed=None):
    drop_remainder = train
    n = next(iter(reviews.values())).shape[0]
    data = {**reviews, 'mask': np.ones(n, dtype=np.bool_)}
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.cache()
    if train:
        ds = ds.shuffle(10000, seed=seed)
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=train)
    if drop_remainder:
        steps_per_epoch = n // batch_size
    else:
        steps_per_epoch = math.ceil(n // batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds, steps_per_epoch


def create_imdb_dataset(data_folder, tokenizer, max_len=512, eval_size=0.2, batch_size=128, eval_batch_size=None,
                        seed=42, with_test=False):
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if with_test:
        train_texts, train_labels = load_imdb_data(os.path.join(data_folder, 'train'))
        test_texts, test_labels = load_imdb_data(os.path.join(data_folder, 'test'))
    else:
        texts, labels = load_imdb_data(data_folder)

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=eval_size, random_state=seed)

    train_reviews = encode_reviews(tokenizer, train_texts, max_len)
    test_reviews = encode_reviews(tokenizer, test_texts, max_len)

    train_data = {'inputs': train_reviews['input_ids'], 'attn_mask': train_reviews['attention_mask'], 'labels': train_labels}
    test_data = {'inputs': test_reviews['input_ids'], 'attn_mask': test_reviews['attention_mask'], 'labels': test_labels}

    def cast_dtype(x):
        x['inputs'] = x['inputs'].astype(np.int32)
        x['attn_mask'] = x['attn_mask'].astype(np.bool_)
        return x

    train_data = cast_dtype(train_data)
    test_data = cast_dtype(test_data)

    ds_train, steps_per_epoch = create_ds(train_data, batch_size, train=True, seed=seed)
    ds_eval, eval_steps = create_ds(test_data, eval_batch_size, train=False, seed=seed)

    return ds_train, steps_per_epoch, ds_eval, eval_steps