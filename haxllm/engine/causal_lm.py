# pyright: reportUnboundVariable=false
from typing import Any, Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

import optax

import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.training import dynamic_scale as dynamic_scale_lib

from haxllm.utils import load_transformer_params, get_metrics, report_params_and_flops, get_sharding, create_mesh, \
    freeze_params_optimizer, spec_from_dataset
from haxllm.pipeline.text_generation import TextGenerationPipeline
from haxllm.gconfig import get_seed, set_gconfig

Array = Any
DType = Any
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


def cast_half(p, dtype):
    if p.dtype == jnp.float32:
        return p.astype(dtype)
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
               model, tx, cast=None, axis_name=None):
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
        params_dtype = jax.tree.map(lambda x: x.dtype, params)
        params = jax.tree.map(lambda p: cast_half(p, cast), params)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (pred_labels, per_example_loss)), grads = grad_fn(params)
    if axis_name is not None:
        grads = jax.lax.pmean(grads, axis_name)

    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    if cast:
        params = jax.tree.map(lambda p, d: p.astype(d), params, params_dtype)

    metrics = compute_metrics(pred_labels, labels, per_example_loss, mask)
    metrics["total"] = mask.sum()
    if axis_name is not None:
        metrics = jax.lax.psum(metrics, axis_name)
    return params, opt_state, step + 1, metrics


def train_step_dynamic_scale(
        params, opt_state, step, dynamic_scale, batch, dropout_rng,
        model, tx, cast=None, axis_name=None):
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
        per_example_loss = causal_lm_loss(logits=logits, labels=labels)
        loss = per_example_loss.mean()
        pred_labels = jnp.argmax(logits, axis=-1)
        return loss, (pred_labels, per_example_loss)

    if cast:
        params_dtype = jax.tree.map(lambda x: x.dtype, params)
        params = jax.tree.map(lambda p: cast_half(p, cast), params)

    grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name=axis_name)
    dynamic_scale, is_fin, (_, (pred_labels, per_example_loss)), grads = grad_fn(params)
    
    # TODO: not sure if this is necessary
    # if cast:
    #     grads = jax.tree.map(lambda g: g.astype(cast), grads)

    def update_fn(grads, params, opt_state):
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Note: no need to use jax.lax.cond here, jit will optimize it.
    new_params, new_opt_state = update_fn(grads, params, opt_state)
    params = jax.tree_util.tree_map(
        partial(jnp.where, is_fin), new_params, params)
    opt_state = jax.tree_util.tree_map(
        partial(jnp.where, is_fin), new_opt_state, opt_state)

    if cast:
        params = jax.tree.map(lambda p, d: p.astype(d), params, params_dtype)

    metrics = compute_metrics(pred_labels, labels, per_example_loss, mask)
    metrics["total"] = mask.sum()
    if axis_name is not None:
        metrics = jax.lax.psum(metrics, axis_name)
    return params, opt_state, step + 1, dynamic_scale, metrics


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
    params = freeze(params)
    opt_state = tx.init(params)
    step = 0
    return params, opt_state, step


def _make_example_inputs(dataset):
    input_spec = spec_from_dataset(dataset, ["input_ids"])
    return {
        key: jnp.ones(spec["shape"], spec["dtype"]) for key, spec in input_spec.items()
    }


class TrainerBase:
    model: nn.Module
    tx: optax.GradientTransformation
    cast: Optional[DType]
    dynamic_scale: bool
    init_scale: float

    _rng: Array
    _params: PyTree
    _opt_state: PyTree
    _global_step: int
    _dynamic_scale: Optional[dynamic_scale_lib.DynamicScale]

    _p_train_step: Callable
    _p_eval_step: Callable

    def init(self, example_inputs, checkpoint=None):
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

    def set_pipeline(
            self, tokenizer, max_len, temperature: float = 1.0, top_k: int = -1,
            top_p: float = 1.0, min_p: float = 0.0, **kwargs):
        model = self.model
        model = model.clone(config=model.config.replace(decode=True, shard_cache=True, padding_left=True))
        pipeline = TextGenerationPipeline(
            tokenizer, model, max_len, rng=self._rng, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, **kwargs)
        pipeline.init_without_params(getattr(self, "mesh", None))
        self.pipeline = pipeline

    @classmethod
    def make_example_inputs(cls, dataset):
        return _make_example_inputs(dataset)

    def load_checkpoint(self, params, checkpoint, device):
        params = unfreeze(flatten_dict(params, sep="."))
        params = load_transformer_params(
            params, checkpoint, device=device, lm_head=True)
        params = freeze(unflatten_dict(params, sep="."))
        return params


class DPTrainer(TrainerBase):

    def __init__(self, model, tx, rng=None, cast=None, dynamic_scale=False):
        if dynamic_scale:
            raise NotImplementedError("Dynamic scale is not supported in DPTrainer.")
        self.model = model
        self.tx = tx
        self.cast = jnp.dtype(cast) if cast is not None else None

        if rng is None:
            rng = jax.random.PRNGKey(get_seed())
        self._rng = rng
    
    def init(self, example_inputs, checkpoint=None):
        input_ids = example_inputs["input_ids"]
        self.replace_freeze_params_optimizer(self._rng, input_ids)
        self._rng, init_rng = jax.random.split(self._rng)

        with jax.default_device(jax.devices("cpu")[0]):
            params, opt_state, global_step = jax.jit(
                partial(init_fn, model=self.model, tx=self.tx),
            )(init_rng, input_ids)

        if checkpoint is not None:
            params = self.load_checkpoint(params, checkpoint, device="cpu")

        params, opt_state, global_step = jax.device_put_replicated(
            (params, opt_state, global_step), jax.local_devices())

        self._params = params
        self._opt_state = opt_state
        self._global_step = global_step

        p_train_step = jax.pmap(
            partial(train_step, model=self.model, tx=self.tx, cast=self.cast, axis_name="batch"),
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


class MPTrainer(TrainerBase):

    def __init__(self, model, tx, rng=None, mesh=(1, 8), cast=None, dynamic_scale=False, init_scale=256.0):
        self.model = model
        self.tx = tx
        self.cast = jnp.dtype(cast) if cast is not None else None
        self.dynamic_scale = dynamic_scale
        self.init_scale = init_scale
        self.mesh = create_mesh(mesh, ("X", "Y"))
        set_gconfig("mesh", self.mesh)

        if rng is None:
            rng = jax.random.PRNGKey(get_seed())
        self._rng = rng
        if self.dynamic_scale:
            scale = jnp.asarray(self.init_scale, dtype="float32")
            dynamic_scale = dynamic_scale_lib.DynamicScale(scale=scale)
            dynamic_scale = jax.device_put(dynamic_scale, NamedSharding(self.mesh, P()))
            self._dynamic_scale = dynamic_scale

    def init(self, example_inputs, checkpoint=None):
        input_ids = example_inputs["input_ids"]
        self.replace_freeze_params_optimizer(self._rng, input_ids)
        self._rng, init_rng = jax.random.split(self._rng)

        out_shardings = get_sharding(
            self.mesh, partial(init_fn, model=self.model, tx=self.tx), init_rng, input_ids
        )

        params, opt_state, global_step = jax.jit(
            partial(init_fn, model=self.model, tx=self.tx),
            out_shardings=out_shardings,
        )(init_rng, input_ids)
        
        if checkpoint is not None:
            params = self.load_checkpoint(params, checkpoint, device=self.mesh)
        
        self._params = params
        self._opt_state = opt_state
        self._global_step = global_step

        none_sharding = out_shardings[-1]
        if self.dynamic_scale:
            p_train_step = jax.jit(
                partial(train_step_dynamic_scale, model=self.model, tx=self.tx, cast=self.cast),
                out_shardings=out_shardings + (none_sharding,) * 2,
                donate_argnums=(0, 1, 2))
        else:
            p_train_step = jax.jit(
                partial(train_step, model=self.model, tx=self.tx, cast=self.cast),
                out_shardings=out_shardings + (none_sharding,),
                donate_argnums=(0, 1, 2))

        p_eval_step = jax.jit(
            partial(eval_step, model=self.model))
        
        self._p_train_step = p_train_step
        self._p_eval_step = p_eval_step
        self._rng, self._dropout_rng = jax.random.split(self._rng)
    
    def train_step(self, batch):
        if self.dynamic_scale:
            self._params, self._opt_state, self._global_step, self._dynamic_scale, metrics = self._p_train_step(
                self._params, self._opt_state, self._global_step, self._dynamic_scale, batch, self._dropout_rng,
            )
        else:
            self._params, self._opt_state, self._global_step, metrics = self._p_train_step(
                self._params, self._opt_state, self._global_step, batch, self._dropout_rng,
            )
        return jax.device_get(metrics)

    def eval_step(self, batch):
        metrics = self._p_eval_step(self._params, batch)
        return jax.device_get(metrics)
    
    def get_metrics(self, metrics_list):
        return get_metrics(metrics_list, pmap=False)

    def predict_step(self, batch):
        output_ids = self.pipeline.random_sample(batch["input_ids"])
        return output_ids
