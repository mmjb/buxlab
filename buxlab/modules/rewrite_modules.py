from abc import ABC
from typing import Sequence, Tuple

from flax.struct import PyTreeNode
import flax.linen as nn
import gin
import jax.numpy as jnp

from .mlp import MLP
from ..utils.misc import aggregate_pytree_leaves


class RewriteScoringMetrics(PyTreeNode):
    loss: float
    num_samples: int
    num_correct: int


class RewriteScoringModule(ABC, nn.Module):
    def compute_metrics(
        self,
        candidate_logprobs: jnp.ndarray,
        correct_candidate_idx: jnp.ndarray,
        correct_is_nonpad: jnp.ndarray,
        selected_fixes: jnp.ndarray,
    ) -> RewriteScoringMetrics:
        """Compute metrics (including loss) for a class of rewrites.

        Args:
            candidate_logprobs: float [num_(tr|vs|as)] tensor, given log-probabilities for each candidate
            correct_candidate_idx: int [num_correct_(tr|vs|as)] tensor, the idxs in candidate_logprobs that are correct
            correct_is_nonpad: bool [num_correct_(tr|vs|as)] tensor, indicates padding in `correct_candidate_idx`.
            selected_fixes: bool [num_(tr|vs|as)] tensor, indicating if a candidate had maximal logprob for the loc
        """
        num_samples = correct_is_nonpad.sum()
        # There are sometimes no samples for a given rewrite type in a batch, so protect from NaNs:
        loss = -(candidate_logprobs[correct_candidate_idx] * correct_is_nonpad).sum() / num_samples
        loss = jnp.nan_to_num(loss, nan=0.0)
        num_correct = (selected_fixes[correct_candidate_idx] * correct_is_nonpad).sum()

        return RewriteScoringMetrics(loss=loss, num_samples=num_samples, num_correct=num_correct)

    @staticmethod
    def aggregate_metrics(
        metrics: Sequence[RewriteScoringMetrics],
    ) -> RewriteScoringMetrics:
        return aggregate_pytree_leaves(metrics)


@gin.configurable
class TextRewriteScoringModule(RewriteScoringModule):
    rewrite_vocab_size: int
    rewrite_embed_size: int = 128
    mlp_dims: Tuple[int, ...] = (128, 64)

    def setup(self):
        self.text_rewrite_embeddings = nn.Embed(
            num_embeddings=self.rewrite_vocab_size, features=self.rewrite_embed_size
        )
        self.text_rewrite_scorer = MLP(dims=self.mlp_dims + (1,))

    def __call__(
        self,
        target_rewrite_node_representations: jnp.ndarray,
        candidate_rewrites: jnp.ndarray,
    ):
        """
        Args:
            target_rewrite_node_representations: [N, D]
            candidate_rewrites: [N]
        """
        embedded_target_rewrites = self.text_rewrite_embeddings(candidate_rewrites)  # [N, D]
        return self.text_rewrite_scorer(
            jnp.concatenate((embedded_target_rewrites, target_rewrite_node_representations), axis=-1)
        ).squeeze(-1)


@gin.configurable
class VarSwapScoringModule(RewriteScoringModule):
    mlp_dims: Tuple[int, ...] = (128, 64)

    def setup(self):
        self.replacement_candidate_scorer = MLP(dims=self.mlp_dims + (1,))

    def __call__(
        self,
        slot_representations_per_target: jnp.ndarray,
        target_nodes_representations: jnp.ndarray,
    ):
        """
        Args:
            slot_representations_per_target:  [N, D]
            target_nodes_representations: [N, D]
        """
        # Compute candidate score by applying MLP to combination of slot and candidate representations:
        return self.replacement_candidate_scorer(
            jnp.concatenate((slot_representations_per_target, target_nodes_representations), axis=-1)
        ).squeeze(-1)


@gin.configurable
class ArgSwapScoringModule(RewriteScoringModule):
    mlp_dims: Tuple[int, ...] = (128, 64)

    def setup(self):
        self.swap_pair_scorer = MLP(dims=self.mlp_dims + (1,))

    def __call__(
        self,
        slot_representations_per_pair: jnp.ndarray,
        pair_representations: jnp.ndarray,
    ):
        """
        Args:
            slot_representations_per_pair: [N, D]
            pair_representations: [N, 2, D]
        """
        # Compute candidate score by applying MLP to combination of slot and pairs representations:
        return self.swap_pair_scorer(
            jnp.concatenate(
                (
                    slot_representations_per_pair,
                    pair_representations.reshape(pair_representations.shape[0], -1),
                ),
                axis=-1,
            )
        ).squeeze(-1)
