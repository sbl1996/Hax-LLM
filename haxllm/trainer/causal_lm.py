# pyright: reportUnboundVariable=false
from typing import Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit

import optax

import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
from flax.traverse_util import flatten_dict, unflatten_dict

from haxllm.utils import load_transformer_params, get_metrics, report_params_and_flops, get_sharding, create_mesh, \
    freeze_params_optimizer


Array = Any
PyTree = Any


def compute_metrics(pred_labels, labels, per_example_loss, mask):
    loss = (per_example_loss * mask).sum()
    acc = causal_lm_accuracy(pred_labels, labels)
    acc = (acc * mask).sum()
    metrics = {
        "loss": loss,
        "acc": acc,
    }
    return metrics


def cast_bf16(p):
    if p.dtype == jnp.float32:
        return p.astype(jnp.bfloat16)
    return p


def causal_lm_loss(logits, labels, ignore_index=-100):
    r"""
    Computes the causal language modeling loss.

    Parameters
    ----------
    logits: jnp.ndarray, shape (batch_size, max_len, vocab_size)
        The logits of the model.
    labels: jnp.ndarray, shape (batch_size, max_len)
        The integer labels of the model.
    ignore_index: int, default -100
        The index to ignore in the loss computation.
    """
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)
    mask = (labels != ignore_index).astype(jnp.float32)
    loss = loss * mask
    loss = loss.sum(axis=1) / mask.sum(axis=1)
    return loss


def causal_lm_accuracy(preds, labels, ignore_index=-100):
    r"""
    Computes the causal language modeling accuracy.

    Parameters
    ----------
    preds: jnp.ndarray, shape (batch_size, max_len)
        The integer predictions of the model.
    labels: jnp.ndarray, shape (batch_size, max_len)
        The integer labels of the model.
    ignore_index: int, default -100
        The index to ignore in the accuracy computation.
    """
    mask = (labels != ignore_index)
    eq = jnp.equal(preds, labels) & mask
    acc = eq.sum(axis=1) / mask.sum(axis=1)
    return acc


def train_step(params, opt_state, step, batch, dropout_rng,
               model, tx, cast=False, axis_name=None):
    labels = batch["labels"]
    mask = batch["mask"]

    dropout_rng = jax.random.fold_in(dropout_rng, step)

    def loss_fn(params):
        logits = model.apply(
            {"params": params},
            input_ids=batch["input_ids"],
            train=True,
            rngs={"dropout": dropout_rng},
        )
        # logits = logits.astype(jnp.float32)
        per_example_loss = causal_lm_loss(logits=logits, labels=labels)
        loss = per_example_loss.mean()
        pred_labels = jnp.argmax(logits, axis=-1)
        return loss, (pred_labels, per_example_loss)

    if cast:
        params_dtype = jax.tree_map(lambda x: x.dtype, params)
        params = jax.tree_map(cast_bf16, params)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (pred_labels, per_example_loss)), grads = grad_fn(params)
    if axis_name is not None:
        grads = jax.lax.pmean(grads, axis_name)

    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    if cast:
        params = jax.tree_map(lambda p, d: p.astype(d), params, params_dtype)

    metrics = compute_metrics(pred_labels, labels, per_example_loss, mask)
    metrics["total"] = mask.sum()
    if axis_name is not None:
        metrics = jax.lax.psum(metrics, axis_name)
    return params, opt_state, step + 1, metrics


def eval_step(params, batch, model, axis_name=None):
    labels = batch["labels"]
    mask = batch["mask"]

    logits = model.apply(
        {"params": params},
        input_ids=batch["input_ids"],
        train=False,
    )
    logits = logits.astype(jnp.float32)
    per_example_loss = causal_lm_loss(logits=logits, labels=labels)
    pred_labels = jnp.argmax(logits, axis=-1)

    metrics = compute_metrics(pred_labels, labels, per_example_loss, mask)
    metrics["total"] = mask.sum()
    if axis_name is not None:
        metrics = jax.lax.psum(metrics, axis_name)
    return metrics


def init_fn(init_rng, input_ids, model, tx):
    params = model.init(init_rng, input_ids=input_ids)["params"]
    opt_state = tx.init(params)
    step = 0
    return params, opt_state, step


class Trainer:
    model: nn.Module
    tx: optax.GradientTransformation
    cast_params_bf16: bool

    _rng: Array
    _params: PyTree
    _opt_state: PyTree
    _global_step: int

    _p_train_step: Callable
    _p_eval_step: Callable

    def init(self, input_ids, checkpoint=None):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def eval_step(self, batch):
        raise NotImplementedError

    def get_metrics(self, metrics_list):
        raise NotImplementedError

    def report_params_and_flops(self, max_len, batch_size):
        return report_params_and_flops(self._params, max_len, batch_size)

    def replace_freeze_params_optimizer(self, rng, input_ids):
        tx = self.tx
        if hasattr(tx, "trainable_pattern") and tx.trainable_pattern is not None:
            abs_params = jax.eval_shape(self.model.init, rng, input_ids=input_ids)['params']
            self.tx = freeze_params_optimizer(tx, abs_params, tx.trainable_pattern)


class DPTrainer(Trainer):

    def __init__(self, model, tx, rng, cast_params_bf16=False):
        self.model = model
        self.tx = tx
        self.cast_params_bf16 = cast_params_bf16

        self._rng = rng
    
    def init(self, input_ids, checkpoint=None):
        self.replace_freeze_params_optimizer(self._rng, input_ids)
        self._rng, init_rng = jax.random.split(self._rng)

        jit_init_fn = jax.jit(
            partial(init_fn, model=self.model, tx=self.tx),
        )
        with jax.default_device(jax.devices("cpu")[0]):
            params, opt_state, global_step = jit_init_fn(
                init_rng, input_ids)
        
        if checkpoint is not None:
            params = unfreeze(flatten_dict(params, sep="."))
            params = load_transformer_params(
                params, checkpoint, device="cpu")
            params = freeze(unflatten_dict(params, sep="."))

        params, opt_state, global_step = jax.device_put_replicated(
            (params, opt_state, global_step), jax.local_devices())

        self._params = params
        self._opt_state = opt_state
        self._global_step = global_step

        p_train_step = jax.pmap(
            partial(train_step, model=self.model, tx=self.tx, cast=self.cast_params_bf16, axis_name="batch"),
            axis_name="batch",
            donate_argnums=(0, 1, 2),
        )
        p_eval_step = jax.pmap(
            partial(eval_step, model=self.model, axis_name="batch"),
            axis_name="batch",
        )

        self._p_train_step = p_train_step
        self._p_eval_step = p_eval_step

        n_devices = jax.local_device_count()
        rngs = jax.random.split(self._rng, n_devices + 1)
        self._rng = rngs[0]
        self._dropout_rng = rngs[1:]

    def _shard(self, batch):
        n_devices = jax.local_device_count()
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((n_devices, -1) + x.shape[1:]), batch)
        return batch

    def train_step(self, batch):
        batch = self._shard(batch)
        self._params, self._opt_state, self._global_step, metrics = self._p_train_step(
            self._params, self._opt_state, self._global_step, batch, self._dropout_rng,
        )
        return jax.device_get(metrics)
    
    def eval_step(self, batch):
        batch = self._shard(batch)
        metrics = self._p_eval_step(self._params, batch)
        return jax.device_get(metrics)

    def get_metrics(self, metrics_list):
        return get_metrics(metrics_list, pmap=True)

    def report_params_and_flops(self, max_len, batch_size):
        return super().report_params_and_flops(max_len, batch_size) // jax.local_device_count()

    def freeze(self, pattern):
        self._params = freeze(self._params, pattern=pattern)


class MPTrainer(Trainer):

    def __init__(self, model, tx, rng, mesh=(1, 8), cast_params_bf16=False):
        self.model = model
        self.tx = tx
        self.cast_params_bf16 = cast_params_bf16
        self.mesh = create_mesh(mesh, ("X", "Y"))
    
        self._rng = rng

    def init(self, input_ids, checkpoint=None):
        self.replace_freeze_params_optimizer(self._rng, input_ids)
        self._rng, init_rng = jax.random.split(self._rng)

        out_shardings = get_sharding(
            self.mesh, partial(init_fn, model=self.model, tx=self.tx), init_rng, input_ids
        )

        jit_init_fn = pjit(
            partial(init_fn, model=self.model, tx=self.tx),
            out_shardings=out_shardings,
        )
        params, opt_state, global_step = jit_init_fn(
            init_rng, input_ids)
        
        if checkpoint is not None:
            params = unfreeze(flatten_dict(params, sep="."))
            params = load_transformer_params(
                params, checkpoint, device=self.mesh)
            params = freeze(unflatten_dict(params, sep="."))
        
        self._params = params
        self._opt_state = opt_state
        self._global_step = global_step

        p_train_step = pjit(
            partial(train_step, model=self.model, tx=self.tx, cast=self.cast_params_bf16),
            out_shardings=out_shardings + out_shardings[-1:],
            donate_argnums=(0, 1, 2))
        p_eval_step = pjit(
            partial(eval_step, model=self.model))
        
        self._p_train_step = p_train_step
        self._p_eval_step = p_eval_step
        self._rng, self._dropout_rng = jax.random.split(self._rng)
    
    def train_step(self, batch):
        self._params, self._opt_state, self._global_step, metrics = self._p_train_step(
            self._params, self._opt_state, self._global_step, batch, self._dropout_rng,
        )
        return jax.device_get(metrics)

    def eval_step(self, batch):
        metrics = self._p_eval_step(self._params, batch)
        return jax.device_get(metrics)
    
    def get_metrics(self, metrics_list):
        return get_metrics(metrics_list, pmap=False)
