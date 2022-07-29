from functools import partial
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Tuple, Protocol, Optional
import time

import gin
import jax
import optax
from absl import app, flags, logging
from flax.metrics import tensorboard
from flax.training import train_state
from flax.core.scope import VariableDict
from flax.linen.module import RNGSequences


from ..modules.rewrite_chooser_module import RewriteChooserBatchLabels, RewriteChooserMetrics
from ..modules.gnn_buxlab import GNNBuxlabBatchFeatures, GNNBuxlabModel
from ..utils.buglab_dataset import BuglabDataset
from ..utils.metric_logger import MetricLogger
from ..utils.misc import aggregate_pytree_leaves

flags.DEFINE_multi_string("gin_file", None, "List of paths to the Gin config files.")
flags.DEFINE_multi_string("gin_param", None, "Newline separated list of Gin parameter bindings.")
flags.DEFINE_string("data_dir", None, "Directory containing data.")
flags.DEFINE_string("result_dir", None, "Directory to store model data.")

FLAGS = flags.FLAGS


class BuxlabModuleApplyFun(Protocol):
    def __call__(
        self,
        params: VariableDict,
        rc_batch: GNNBuxlabBatchFeatures,
        rc_labels: RewriteChooserBatchLabels,
        train: bool,
        train_step: int,
        rngs: Optional[RNGSequences],
    ) -> Tuple[float, RewriteChooserMetrics]:
        ...


def run_and_eval(
    model_params,
    apply_fn: BuxlabModuleApplyFun,
    batch_features: GNNBuxlabBatchFeatures,
    batch_labels: RewriteChooserBatchLabels,
    train_step: int,
    rngs: Optional[RNGSequences],
    train: bool = False,
) -> Tuple[float, RewriteChooserMetrics]:
    return apply_fn(
        {"params": model_params},
        batch_features,
        batch_labels,
        train=train,
        train_step=train_step,
        rngs=rngs,
    )


@partial(jax.jit, static_argnames=["apply_fn"])  # type: ignore
def run_and_eval_fast(
    model_params,
    apply_fn: BuxlabModuleApplyFun,
    batch_features: GNNBuxlabBatchFeatures,
    batch_labels: RewriteChooserBatchLabels,
    train_step: int,
    rngs: Optional[RNGSequences],
    train: bool = False,
) -> Tuple[float, RewriteChooserMetrics]:
    return run_and_eval(model_params, apply_fn, batch_features, batch_labels, train_step, rngs, train)


@partial(jax.jit, static_argnames=["apply_fn"])  # type: ignore
def train_step(
    state: train_state.TrainState,
    apply_fn: BuxlabModuleApplyFun,
    batch_features: GNNBuxlabBatchFeatures,
    batch_labels: RewriteChooserBatchLabels,
    rngs: Optional[RNGSequences],
) -> Tuple[train_state.TrainState, RewriteChooserMetrics]:
    """Computes gradients, loss and accuracy for a single batch."""
    grad_fn = jax.value_and_grad(run_and_eval, has_aux=True)
    (_, metrics), grads = grad_fn(
        state.params,
        apply_fn,
        batch_features,
        batch_labels,
        train_step=state.step,
        rngs=rngs,
        train=True,
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


def run_eval(
    state: train_state.TrainState,
    apply_fn: BuxlabModuleApplyFun,
    data_iter: Iterator[Tuple[GNNBuxlabBatchFeatures, RewriteChooserBatchLabels]],
) -> RewriteChooserMetrics:
    epoch_metrics: List[RewriteChooserMetrics] = []
    for batch_features, batch_labels in data_iter:
        _, batch_metrics = run_and_eval_fast(
            state.params,
            apply_fn,
            batch_features,
            batch_labels,
            train_step=state.step,
            rngs={},
        )
        print(compute_accs(batch_metrics))
        epoch_metrics.append(batch_metrics)
    return aggregate_pytree_leaves(epoch_metrics)


def compute_accs(metrics: RewriteChooserMetrics) -> Dict[str, float]:
    return {
        "loc_acc": metrics.localization.num_correct / metrics.localization.num_samples,
        "tr_acc": metrics.text_rewrite_scoring.num_correct / metrics.text_rewrite_scoring.num_samples,
        "vs_acc": metrics.varswap_scoring.num_correct / metrics.varswap_scoring.num_samples,
        "as_acc": metrics.argswap_scoring.num_correct / metrics.argswap_scoring.num_samples,
    }


@gin.configurable(denylist=["model", "dataset", "initial_state", "summary_writer", "seed"])
def train_loop(
    model: GNNBuxlabModel,
    dataset: BuglabDataset,
    initial_state: train_state.TrainState,
    summary_writer: tensorboard.SummaryWriter,
    num_steps: int = 1000,
    log_every_num_steps: int = 25,
    eval_every_num_steps: int = 10,
    patience: int = 25,
    seed: int = 0,
):
    module = model.build_module()
    train_logger = MetricLogger(log_every_num_steps, "train/", summary_writer)
    valid_logger = MetricLogger(1, "valid/", summary_writer)

    train_samples = dataset.get_sample_iterable("train", seed=seed, repeat=True)
    train_batch_it = model.batch_iterator(train_samples)
    valid_samples = dataset.get_sample_iterable("valid", seed=seed, repeat=False)

    rng = jax.random.PRNGKey(seed)
    state = initial_state
    best_valid_loc_acc, best_state, evals_without_improv = 0, initial_state, 0

    apply_fn: BuxlabModuleApplyFun = partial(module.apply)  # type: ignore

    for step in range(num_steps):
        if step % eval_every_num_steps == 0:
            metrics = run_eval(state, apply_fn, model.batch_iterator(valid_samples))
            accs = compute_accs(metrics)
            valid_logger.log(**accs)

            if accs["loc_acc"] > best_valid_loc_acc:
                logging.info("  (best so far)")
                best_valid_loc_acc = accs["loc_acc"]
                best_state = state
                evals_without_improv = 0
            else:
                evals_without_improv += 1
                if evals_without_improv > patience:
                    logging.info(
                        f"Stopping training after {patience} epochs without valid localization acc. improvement."
                    )
                    break

        batch_features, batch_labels = next(train_batch_it)
        dropout_rng, rng = jax.random.split(rng)
        start_time = time.time()
        state, metrics = train_step(state, apply_fn, batch_features, batch_labels, rngs={"dropout": dropout_rng})
        duration = time.time() - start_time
        logging.info(f"Step {step}: {metrics.loss}, {duration}s")
        train_logger.log(loss=metrics.loss, time=duration, **compute_accs(metrics))

    return best_state


@gin.configurable
def create_optimizer(
    learning_rate: float = 0.001,
    warmup_steps: int = 100,
    cosine_steps: int = 400,
    cosine_restarts: int = 5,
    cosine_alpha: float = 0.1,
    grad_clip_bound: Optional[float] = 1.0,
    optimizer_type: Literal["sgd", "adam", "adamw"] = "adamw",
) -> optax.GradientTransformation:
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=cosine_steps,
        alpha=cosine_alpha,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn] + [cosine_fn] * (cosine_restarts + 1),
        boundaries=[warmup_steps] + [warmup_steps + i * cosine_steps for i in range(1, cosine_restarts)],
    )

    opt_chain = []
    if grad_clip_bound is not None:
        opt_chain.append(optax.clip(grad_clip_bound))

    if optimizer_type == "sgd":
        opt_chain.append(optax.sgd(learning_rate=schedule_fn))
    elif optimizer_type == "adam":
        opt_chain.append(optax.adam(learning_rate=schedule_fn))
    elif optimizer_type == "adamw":
        opt_chain.append(optax.adamw(learning_rate=schedule_fn))
    else:
        raise ValueError(f"Unknown optimizer type {optimizer_type}")

    return optax.chain(*opt_chain)


@gin.configurable(denylist=("data_dir", "result_dir"))
def train(
    data_dir: str,
    result_dir: str,
    seed: int = 0,
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      data_dir: Directory with the training, validation and test data.
      result_dir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    rng = jax.random.PRNGKey(seed)
    dataset = BuglabDataset(data_folder=Path(data_dir))
    summary_writer = tensorboard.SummaryWriter(result_dir)

    model = GNNBuxlabModel()
    model.compute_metadata(dataset.get_sample_iterable("train", seed=0, repeat=False))

    # Create initial model parameters:
    train_samples = dataset.get_sample_iterable("train", seed=seed, repeat=False)
    sample_features, sample_labels = next(model.batch_iterator(train_samples))
    module = model.build_module()
    params = module.init(rng, sample_features, sample_labels)["params"]
    init_state = train_state.TrainState.create(apply_fn=module.apply, params=params, tx=create_optimizer())

    logging.info("Model params:\n" + str(jax.tree_util.tree_map(lambda x: x.shape, init_state.params)))

    trained_state = train_loop(
        model,
        dataset,
        init_state,
        summary_writer,
        seed=seed,
    )

    summary_writer.flush()

    # _, test_f1 = run_eval(trained_state, iter(test_ds))
    # print("test_f1: %.3f" % (test_f1,))

    return trained_state


def main(argv):
    gin.external_configurable(optax.sgd)
    gin.external_configurable(optax.adam)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    logging.info("Config:\n" + gin.config_str())

    # TODO: log hparams from gin to tensorboard.SummaryWriter.hparams

    train(FLAGS.data_dir, FLAGS.result_dir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["gin_file", "data_dir", "result_dir"])

    try:
        app.run(main)
    except:
        import sys, traceback, pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
