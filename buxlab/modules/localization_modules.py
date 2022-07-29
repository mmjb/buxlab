from typing import Callable, Sequence
import math

from abc import ABC, abstractmethod
from flax.struct import PyTreeNode
import flax.linen as nn
import gin
import jax.numpy as jnp

from ..utils.misc import aggregate_pytree_leaves
from ..utils.segment_ops import segment_max, segment_log_softmax


class LocalizationMetrics(PyTreeNode):
    num_samples: int
    num_correct: int
    num_no_bug: int
    num_no_bug_correct: int
    loss: float


class LocalizationModule(nn.Module, ABC):
    buggy_samples_weight_schedule: Callable[[int], float] = lambda _: 1.0
    abstain_weight: float = 0.0

    @abstractmethod
    def compute_localization_logprobs(
        self,
        candidate_reprs: jnp.ndarray,
        candidate_to_sample_idx: jnp.ndarray,
        num_samples: int,
    ) -> jnp.ndarray:
        """
        Args:
            candidate_reprs: [num_candidates, H], representation of each rewritable location.
            candidate_to_sample_idx: [num_candidates], index tensor, mapping each candidate to a sample.
                (Needed to do a softmax)
            num_samples: int scalar, number of samples in our input

        Returns:
            float array of shape [num_candidates + num_samples], such that
        """
        raise NotImplementedError()

    def compute_metrics(
        self,
        candidate_log_probs: jnp.ndarray,
        candidate_to_sample_idx: jnp.ndarray,
        sample_has_bug: jnp.ndarray,
        sample_to_correct_candidate_idx: jnp.ndarray,
        sample_is_nonpad: jnp.ndarray,
        train_step: int,
    ) -> LocalizationMetrics:
        """Compute metrics (including loss) for localization.

        Args:
            candidate_log_probs: [num_candidates + num_samples], float tensor of log prob per candidate
            candidate_to_sample_idx: [num_candidates], int tensor mapping each candidate to a sample
            sample_has_bug: [num_samples], bool tensor indicating whether a sample has a bug
            sample_to_correct_candidate_idx: [num_samples], int tensor indicating the correct candidate for each sample.
                Garbage values for non-buggy samples.
            sample_is_nonpad: [num_samples], bool tensor indicating whether a sample is padding.
            train_step: int, indicating where we are in training. Used to determine loss weights.
        """
        # candidate_log_probs is made of two parts - actual candidates, and then "virtual" NoBug locations.
        # We need to treat these differently from time to time...
        num_candidates = candidate_to_sample_idx.shape[0]
        num_samples = sample_has_bug.shape[0]
        num_nonpad_samples = sample_is_nonpad.sum()
        sample_range = jnp.arange(num_samples)
        no_bug_indices = num_candidates + sample_range
        candidate_to_sample_idx = jnp.concatenate((candidate_to_sample_idx, sample_range))

        # The input sample_to_correct_candidate_idx has garbage for samples that are not buggy; replace those values:
        sample_to_correct_candidate_idx = jnp.where(
            sample_has_bug, sample_to_correct_candidate_idx, no_bug_indices
        )  # [num_samples]

        log_probs_per_sample: jnp.ndarray = jnp.where(
            sample_is_nonpad, candidate_log_probs[sample_to_correct_candidate_idx], 0.0
        )
        log_probs_per_sample = log_probs_per_sample.clip(-math.inf, math.log(0.995))
        if self.abstain_weight > 0:
            log_probs_per_sample = log_probs_per_sample + jnp.where(
                sample_has_bug * sample_is_nonpad,
                self.abstain_weight * candidate_log_probs[no_bug_indices],
                jnp.zeros_like(log_probs_per_sample),
            )

        buggy_samples_weight = self.buggy_samples_weight_schedule(train_step)
        if buggy_samples_weight == 1.0:
            loss = -log_probs_per_sample.sum() / num_nonpad_samples
        else:
            weights = jnp.where(
                sample_has_bug,
                jnp.full_like(log_probs_per_sample, fill_value=buggy_samples_weight),
                jnp.full_like(log_probs_per_sample, fill_value=1.0),
            )
            loss = -(log_probs_per_sample * weights * sample_is_nonpad).sum() / (weights * sample_is_nonpad).sum()

        # TODO: currently jax does not support a sparse argmax op (see https://github.com/google/jax/issues/10079)
        # So instead, we check if the computed max for each segment coincides with the extracted logprobs:
        max_log_prob_per_sample = segment_max(
            data=candidate_log_probs,
            segment_ids=candidate_to_sample_idx,
            num_segments=num_samples,
        )
        correct_samples = (max_log_prob_per_sample == log_probs_per_sample) * sample_is_nonpad
        sample_has_no_bug = jnp.logical_not(sample_has_bug) * sample_is_nonpad

        return LocalizationMetrics(
            loss=loss,
            num_samples=num_nonpad_samples,
            num_correct=correct_samples.sum(),
            num_no_bug=sample_has_no_bug.sum(),
            num_no_bug_correct=(sample_has_no_bug * correct_samples).sum(),
        )

    @staticmethod
    def aggregate_metrics(
        metrics: Sequence[LocalizationMetrics],
    ) -> LocalizationMetrics:
        return aggregate_pytree_leaves(metrics)


@gin.configurable
class CandidateQueryPointerNetLocalizationModule(LocalizationModule):
    hidden_dim: int = 64

    def setup(self):
        self.to_sample_repr = nn.Dense(features=self.hidden_dim, use_bias=False)
        self.to_hidden_candidate_repr = nn.Dense(features=self.hidden_dim, use_bias=False)
        self.to_candidate_score = nn.Dense(features=1, use_bias=False)

    def compute_localization_logprobs(self, candidate_reprs, candidate_to_sample_idx, num_samples):
        num_candidates = candidate_reprs.shape[0]
        sample_repr = segment_max(
            data=self.to_sample_repr(candidate_reprs),
            segment_ids=candidate_to_sample_idx,
            num_segments=num_samples,
        )[
            candidate_to_sample_idx
        ]  # [num_candidates, D] - for each candidate, the element-wise max across all candidates for the same sample
        hidden_candidate_repr = nn.sigmoid(
            self.to_hidden_candidate_repr(jnp.concatenate((candidate_reprs, sample_repr), axis=-1))
        )
        candidate_scores = self.to_candidate_score(hidden_candidate_repr).squeeze(-1)  # [num_candidates]

        arange = jnp.arange(num_samples)
        candidate_scores_with_no_bug = jnp.concatenate(
            (
                candidate_scores,
                jnp.ones_like(arange, dtype=jnp.float32),
            )
        )  # [num_candidates + num_samples]
        candidate_to_sample_idx = jnp.concatenate((candidate_to_sample_idx, arange))  # [num_candidates + num_samples]
        candidate_log_probs = segment_log_softmax(
            logits=candidate_scores_with_no_bug,
            segment_ids=candidate_to_sample_idx,
            num_segments=num_candidates + num_samples,
        )  # [num_candidates + num_samples]

        return candidate_log_probs
