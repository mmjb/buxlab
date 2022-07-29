from typing import Tuple, Callable, Type

from flax import linen as nn
from flax.struct import PyTreeNode
import logging
import jax.numpy as jnp


from .rewrite_modules import (
    ArgSwapScoringModule,
    RewriteScoringMetrics,
    TextRewriteScoringModule,
    VarSwapScoringModule,
)
from .localization_modules import LocalizationMetrics, LocalizationModule
from ..utils.segment_ops import segment_log_softmax, segment_max

LOGGER = logging.getLogger(__name__)


class RewriteChooserBatchFeatures(PyTreeNode):
    """Struct to hold batched information for rewriter chooser modules,
    abstracting away from chosen representation model.

    Used shape abbreviations:
        D ~ hidden dimension of core code embedding model
        num_tr ~ number of text rewrite candidates in batch
        num_vs ~ number of var swap candidates in batch
        num_as ~ number of arg swap candidates in batch
        num_cands ~ number of rewrite candidates in batch (= num_tr + num_vs + num_as)
        num_locs ~ number of rewritable locations in batch (< num_cands, as there are sometimes
            several candidates for a single location)
        num_samples ~ number of samples in batch (< num_locs, as each sample may have several locations)

    Attributes:
        sample_is_nonpad: optional bool [num_samples] tensor, indicating which entries of `sample_has_bug` and
            `sample_to_correct_loc_idx` are not padding.
        rewritable_loc_to_sample_id: int [num_locs] tensor, range [0, num_samples) - indicates which sample a potential
            rewrite location belongs to.

        text_rewrite_replacement_ids: int [num_tr] tensor - IDs in rewrite vocabulary of the candidate rewrites.
        text_rewrite_to_loc_group: int [num_tr] tensor - maps each rewrite to a batch-specific
            index for all rewrites referring to the same location.

        varswap_to_loc_group: int [num_vs] tensor - maps each variable to a batch-specific
            index for all rewrites referring to the same location.

        argswap_to_loc_group: int [num_as] tensor - maps each argument location to a
            batch-specific index for all rewrites referring to the same location.
    """

    sample_is_nonpad: jnp.ndarray

    # Localization information
    rewritable_loc_to_sample_id: jnp.ndarray

    # Text rewrites
    text_rewrite_replacement_ids: jnp.ndarray
    text_rewrite_to_loc_group: jnp.ndarray

    # Var Misuse
    varswap_to_loc_group: jnp.ndarray

    # Arg swaps
    argswap_to_loc_group: jnp.ndarray


class RewriteChooserBatchLabels(PyTreeNode):
    """Struct to hold batched label information for rewriter chooser modules.

    Attributes:
        sample_has_bug: optional bool [num_samples] tensor, indicating whether the sample is buggy
        sample_to_correct_loc_idx: optional int [num_samples] tensor, range [0, num_locs] + {-10000}, indicates the
            index of the the rewritable location that is buggy (-10000 for "not buggy")

        text_rewrite_correct_idxs: optional int [num_correct_tr] tensor - indices into [0, num_tr-1]
            of those text rewrites that were correct.
        text_rewrite_correct_is_nonpad: optional bool [num_correct_tr] tensor - indicates which entries of
            `text_rewrite_correct_idxs` are not padding.

        varswap_correct_idxs: optional int [num_correct_vs] tensor - indices into [0, num_vs-1]
            of those var swaps that were correct.
        varswap_correct_is_nonpad: optional bool [num_correct_vs] tensor - indicates which entries of
            `varswap_correct_idxs` are not padding.

        argswap_correct_idxs: optional int [num_correct_as] tensor - indices into [0, num_as-1]
            of those arg swaps that were correct.
        argswap_correct_is_nonpad: optional bool [num_correct_as] tensor - indicates which entries of
            `argswap_correct_is_nonpad` are not padding.
    """

    # Localization information
    sample_has_bug: jnp.ndarray
    sample_to_correct_loc_idx: jnp.ndarray

    # Text rewrites
    text_rewrite_correct_idxs: jnp.ndarray
    text_rewrite_correct_is_nonpad: jnp.ndarray

    # Var Misuse
    varswap_correct_idxs: jnp.ndarray
    varswap_correct_is_nonpad: jnp.ndarray

    # Arg swaps
    argswap_correct_idxs: jnp.ndarray
    argswap_correct_is_nonpad: jnp.ndarray


class RewriteChooserCodeRepresentations(PyTreeNode):
    """Struct to hold intermediate information for BugLab modules, abstracting away
    from details on how representations are computed and indexed.

    Used shape abbreviations:
        D: hidden dimension of core code embedding model
        num_tr: number of text rewrite candidates in batch
        num_vs: number of var swap candidates in batch
        num_as: number of arg swap candidates in batch
        num_cands: number of rewrite candidates in batch (= num_tr + num_vs + num_as)
        num_locs: number of rewritable locations in batch (< num_cands, as there are often
            several candidates for a single location)
        num_samples: number of samples in batch (< num_locs, as each sample may have several
            rewritable locations)

    Attributes:
        rewritable_loc_reprs: float [num_locs, D] tensor - representation of rewritable locations.

        text_rewrite_loc_reprs: float [num_tr, D] tensor - representations of locations
            at wich text rewrites may be applied (a location may appear several times, once
            per candidate rewrite).

        varswap_loc_reprs: float [num_vs, D] tensor - representations of locations
            at which variable swaps may be applied (a location may appear several times, once
            per candidate replacement variable).
        varswap_replacement_reprs: float [num_vs, D] tensor - representations of
            variables which may be used as replacements, such that the i-th entry represents
            a var that could replace the one represented by varswap_loc_reprs[i].

        argswap_loc_reprs: float [num_as, D] tensor - representations of locations
            at which argument swaps may be applied (a location may appear several times, once
            per candidate replacement argument).
        argswap_swapped_pair_reprs: float [num_as, 2, D] tensor - representations of
            arguments which may be used as replacements, such that the i-th entry represents
            an arg that could replace the one represented by argswap_loc_reprs[i].
    """

    # Localization-specific information:
    rewritable_loc_reprs: jnp.ndarray

    # Text repair-specific information:
    text_rewrite_loc_reprs: jnp.ndarray

    # VarSwap-specific information:
    varswap_loc_reprs: jnp.ndarray
    varswap_replacement_reprs: jnp.ndarray

    # ArgSwap-specific information:
    argswap_loc_reprs: jnp.ndarray
    argswap_swapped_pair_reprs: jnp.ndarray


class RewriteLogprobs(PyTreeNode):
    """Struct to hold the outputs of a Buxlab model.

    Attributes:
        localization_logprobs: float [num_locs + num_samples] tensor - log probabilities of considered
            locations. The final num_samples entries are the virtual NoBug locations.
        text_rewrite_logprobs: float [num_tr] tensor - log probabilities of the text rewrites.
        varswap_logprobs: float [num_vs] tensor - log probabilities of the variable swaps.
        argswap_logprobs: float [num_as] tensor - log probabilities of the argument swaps.
    """

    localization_logprobs: jnp.ndarray

    text_rewrite_logprobs: jnp.ndarray
    varswap_logprobs: jnp.ndarray
    argswap_logprobs: jnp.ndarray


class RewriteChooserMetrics(PyTreeNode):
    loss: float
    num_samples: int

    localization: LocalizationMetrics
    text_rewrite_scoring: RewriteScoringMetrics
    varswap_scoring: RewriteScoringMetrics
    argswap_scoring: RewriteScoringMetrics


class RewriteChooserModule(nn.Module):
    rewrite_vocab_size: int
    localization_module_type: Type[LocalizationModule]
    buggy_samples_weight_schedule: Callable[[int], float]
    repair_weight_schedule: Callable[[int], float]

    def setup(self):
        self.localization_module = self.localization_module_type()
        self.text_rewrite_scoring_module = TextRewriteScoringModule(rewrite_vocab_size=self.rewrite_vocab_size)
        self.varswap_scoring_module = VarSwapScoringModule()
        self.argswap_scoring_module = ArgSwapScoringModule()

    def _compute_rewrite_logprobs(
        self,
        rc_batch: RewriteChooserBatchFeatures,
        rc_codereprs: RewriteChooserCodeRepresentations,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        text_repair_logits = self.text_rewrite_scoring_module(
            target_rewrite_node_representations=rc_codereprs.text_rewrite_loc_reprs,
            candidate_rewrites=rc_batch.text_rewrite_replacement_ids,
        )

        varswap_logits = self.varswap_scoring_module(
            slot_representations_per_target=rc_codereprs.varswap_loc_reprs,
            target_nodes_representations=rc_codereprs.varswap_replacement_reprs,
        )

        argswap_logits = self.argswap_scoring_module(
            slot_representations_per_pair=rc_codereprs.argswap_loc_reprs,
            pair_representations=rc_codereprs.argswap_swapped_pair_reprs,
        )

        # Now compute softmaxes over all logits associated with the same location,
        # and then split it up into logprobs for the individual localization modules again.
        all_logits = jnp.concatenate((text_repair_logits, varswap_logits, argswap_logits))
        logit_groups = jnp.concatenate(
            (
                rc_batch.text_rewrite_to_loc_group,
                rc_batch.varswap_to_loc_group,
                rc_batch.argswap_to_loc_group,
            )
        )
        num_locs = rc_batch.rewritable_loc_to_sample_id.shape[0]
        logprobs = segment_log_softmax(logits=all_logits, segment_ids=logit_groups, num_segments=num_locs)

        text_repair_logprobs = logprobs[: text_repair_logits.shape[0]]
        varswap_logprobs = logprobs[text_repair_logits.shape[0] : text_repair_logits.shape[0] + varswap_logits.shape[0]]
        argswap_logprobs = logprobs[text_repair_logits.shape[0] + varswap_logits.shape[0] :]
        return argswap_logprobs, text_repair_logprobs, varswap_logprobs

    def _compute_rewrite_choice_logprobs_from_reprs(
        self,
        rc_batch: RewriteChooserBatchFeatures,
        rc_codereprs: RewriteChooserCodeRepresentations,
    ) -> RewriteLogprobs:
        loc_logprobs = self.localization_module.compute_localization_logprobs(
            candidate_reprs=rc_codereprs.rewritable_loc_reprs,
            candidate_to_sample_idx=rc_batch.rewritable_loc_to_sample_id,
            num_samples=rc_batch.sample_is_nonpad.shape[0],
        )

        # Get logprobs for repair rewrites conditional on a target location:
        (
            argswap_logprobs,
            text_repair_logprobs,
            varswap_logprobs,
        ) = self._compute_rewrite_logprobs(rc_batch, rc_codereprs)

        return RewriteLogprobs(
            localization_logprobs=loc_logprobs,
            text_rewrite_logprobs=text_repair_logprobs,
            varswap_logprobs=varswap_logprobs,
            argswap_logprobs=argswap_logprobs,
        )

    def compute_detector_metrics(
        self,
        rc_batch: RewriteChooserBatchFeatures,
        rc_labels: RewriteChooserBatchLabels,
        rewrite_logprobs: RewriteLogprobs,
        train_step: int,
    ) -> Tuple[float, RewriteChooserMetrics]:
        """
        Arguments:
            rc_batch: struct holding the input information about this batch.
            rc_codereprs: struct holding the code representations for this batch.
            rewrite_logprobs: struct holding information about chosen rewrites.
            train_step: int, indicating where we are in training. Used to determine loss weights.
        """
        num_locs = rc_batch.rewritable_loc_to_sample_id.shape[0]
        num_tr = rc_batch.text_rewrite_to_loc_group.shape[0]
        num_vs = rc_batch.varswap_to_loc_group.shape[0]
        num_samples = rc_batch.sample_is_nonpad.shape[0]

        localization_metrics: LocalizationMetrics = self.localization_module.compute_metrics(
            candidate_log_probs=rewrite_logprobs.localization_logprobs,
            candidate_to_sample_idx=rc_batch.rewritable_loc_to_sample_id,
            sample_has_bug=rc_labels.sample_has_bug,
            sample_to_correct_candidate_idx=rc_labels.sample_to_correct_loc_idx,
            sample_is_nonpad=rc_batch.sample_is_nonpad,
            train_step=train_step,
        )

        all_logprobs = jnp.concatenate(
            (
                rewrite_logprobs.text_rewrite_logprobs,
                rewrite_logprobs.varswap_logprobs,
                rewrite_logprobs.argswap_logprobs,
            )
        )
        logprob_groups = jnp.concatenate(
            (
                rc_batch.text_rewrite_to_loc_group,
                rc_batch.varswap_to_loc_group,
                rc_batch.argswap_to_loc_group,
            )
        )
        max_logprob_per_group = segment_max(data=all_logprobs, segment_ids=logprob_groups, num_segments=num_locs)
        max_logprob_per_rewrite = max_logprob_per_group[logprob_groups]
        rewrite_is_selected = all_logprobs == max_logprob_per_rewrite

        text_rewrite_is_selected_fix = rewrite_is_selected[:num_tr]
        varswap_is_selected_fix = rewrite_is_selected[num_tr : num_tr + num_vs]
        argswap_is_selected_fix = rewrite_is_selected[num_tr + num_vs :]

        text_rewrite_metrics = self.text_rewrite_scoring_module.compute_metrics(
            rewrite_logprobs.text_rewrite_logprobs,
            rc_labels.text_rewrite_correct_idxs,
            rc_labels.text_rewrite_correct_is_nonpad,
            text_rewrite_is_selected_fix,
        )
        varswap_metrics = self.varswap_scoring_module.compute_metrics(
            rewrite_logprobs.varswap_logprobs,
            rc_labels.varswap_correct_idxs,
            rc_labels.varswap_correct_is_nonpad,
            varswap_is_selected_fix,
        )
        argswap_metrics = self.argswap_scoring_module.compute_metrics(
            rewrite_logprobs.argswap_logprobs,
            rc_labels.argswap_correct_idxs,
            rc_labels.argswap_correct_is_nonpad,
            argswap_is_selected_fix,
        )

        buggy_samples_weight = self.buggy_samples_weight_schedule(train_step)
        repair_weight = self.repair_weight_schedule(train_step)
        repair_loss = text_rewrite_metrics.loss + varswap_metrics.loss + argswap_metrics.loss
        repair_loss = repair_loss * buggy_samples_weight

        weighted_loss = localization_metrics.loss + repair_weight * (repair_loss / num_samples)

        return (
            weighted_loss,
            RewriteChooserMetrics(
                num_samples=num_samples,
                loss=weighted_loss,
                localization=localization_metrics,
                text_rewrite_scoring=text_rewrite_metrics,
                varswap_scoring=varswap_metrics,
                argswap_scoring=argswap_metrics,
            ),
        )
