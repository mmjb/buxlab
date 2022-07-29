from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from flax.struct import PyTreeNode
import flax.linen as nn
import gin
import jax.numpy as jnp

from buxlab.modules.localization_modules import (
    CandidateQueryPointerNetLocalizationModule,
    LocalizationModule,
)

from .rewrite_chooser_module import (
    RewriteChooserMetrics,
    RewriteChooserModule,
    RewriteChooserBatchFeatures,
    RewriteChooserBatchLabels,
    RewriteChooserCodeRepresentations,
    RewriteLogprobs,
)
from .gnn import GNNBatch, GNNModel, GNNModule
from ..utils.buxlab_interface import AbstractBuxlabModel
from ..utils.language_rewrite_vocabs import get_language_vocab
from ..utils.schedules import make_weight_schedule


class GNNBuxlabBatchFeatures(RewriteChooserBatchFeatures):
    """Struct to hold batched information for the GNN Buxlab module. For used shape abbreviations,
    see BuxlabBatch.

    Attributes:
        graph_data: graph representing all samples in this batch
        text_rewrite_node_idxs: int [num_tr] tensor - indices into the reprs produced for `graph_data`,
            for locations where we may want to rewrite a literal
        varswap_node_idxs: int [num_vs] tensor - indices into the reprs produced for `graph_data`,
            indicating a variable that we may want to replace
        varswap_replacement_node_idxs: int [num_vs] tensor - indices into the reprs produced for `graph_data`,
            indicating a replacement for the corresponding variable in `varswap_node_idxs`
        argswap_node_idxs: int [num_as] tensor - indices into the reprs produced for `graph_data`,
            indicating a call node in which we may want to swap arguments
        argswap_swapped_node_pair_idxs: int [num_as, 2] tensor - indices into the reprs produced for `graph_data`,
            indicating pairs of arguments that we may want to swap
    """

    graph_data: GNNBatch
    rewritable_loc_node_idxs: jnp.ndarray
    text_rewrite_node_idxs: jnp.ndarray
    varswap_node_idxs: jnp.ndarray
    varswap_replacement_node_idxs: jnp.ndarray
    argswap_node_idxs: jnp.ndarray
    argswap_swapped_node_pair_idxs: jnp.ndarray


class GNNBuxlabModule(RewriteChooserModule):
    gnn_module_maker: Callable[[], GNNModule]
    use_all_gnn_layer_outputs: bool

    def setup(self):
        super().setup()

        self.gnn = self.gnn_module_maker()
        if self.use_all_gnn_layer_outputs:
            self.summarization_layer = nn.Dense(features=self.gnn.hidden_dim, use_bias=False)
        else:
            self.summarization_layer = None

    def _compute_rewrite_choice_logprobs(
        self, rc_batch: GNNBuxlabBatchFeatures, train: bool = False
    ) -> RewriteLogprobs:
        gnn_output: List[jnp.ndarray] = self.gnn(input=rc_batch.graph_data, train=train)
        if self.summarization_layer is not None:
            node_representations = self.summarization_layer(jnp.concatenate(gnn_output, axis=-1))
        else:
            node_representations = gnn_output[-1]

        return self._compute_rewrite_choice_logprobs_from_reprs(
            rc_batch=rc_batch,
            rc_codereprs=RewriteChooserCodeRepresentations(
                rewritable_loc_reprs=node_representations[rc_batch.rewritable_loc_node_idxs],
                text_rewrite_loc_reprs=node_representations[rc_batch.text_rewrite_node_idxs],
                varswap_loc_reprs=node_representations[rc_batch.varswap_node_idxs],
                varswap_replacement_reprs=node_representations[rc_batch.varswap_replacement_node_idxs],
                argswap_loc_reprs=node_representations[rc_batch.argswap_node_idxs],
                argswap_swapped_pair_reprs=node_representations[rc_batch.argswap_swapped_node_pair_idxs],
            ),
        )

    def __call__(
        self,
        rc_batch: GNNBuxlabBatchFeatures,
        rc_labels: RewriteChooserBatchLabels,
        train_step: int = 0,
        train: bool = False,
    ) -> Tuple[float, RewriteChooserMetrics]:
        rewrite_logprobs = self._compute_rewrite_choice_logprobs(rc_batch, train=train)
        return self.compute_detector_metrics(rc_batch, rc_labels, rewrite_logprobs, train_step)


class RewriteModuleSelectionInformation(PyTreeNode):
    # The nodes at which we can rewrite:
    node_idxs: List[int]
    # The potential replacements (aligned with `node_idxs`); can be IDs into vocab or other nodes
    replacement_ids: List[Union[int, jnp.ndarray]]
    # Grouping index; same for all rewrites at same location (aligned with `node_idxs`):
    loc_groups: List[int]
    # The index of the correct choice in a given location group (if any such exists)
    correct_choice_idx: Optional[int]
    # The original index of the rewrite (in the set of rewrites for all modules)
    original_idxs: List[int]


@gin.configurable
@dataclass
class GNNBuxlabModel(
    AbstractBuxlabModel[
        Dict[str, Any],
        Tuple[GNNBuxlabBatchFeatures, RewriteChooserBatchLabels],
        GNNBuxlabModule,
    ]
):
    gnn_model: GNNModel = GNNModel()
    localization_module_type: Type[LocalizationModule] = CandidateQueryPointerNetLocalizationModule
    use_all_gnn_layer_outputs: bool = True
    rewrite_vocabulary_name: str = "python"
    buggy_samples_weight_schedule_spec: Union[str, float] = 1.0
    repair_weight_schedule_spec: Union[str, float] = 1.0

    def __post_init__(self):
        self.rewrite_vocabulary = get_language_vocab(self.rewrite_vocabulary_name)
        self.buggy_samples_weight_schedule = make_weight_schedule(self.buggy_samples_weight_schedule_spec)
        self.repair_weight_schedule = make_weight_schedule(self.repair_weight_schedule_spec)

    def metadata_init(self) -> Dict[str, Any]:
        return {
            "gnn_metadata": self.gnn_model.metadata_init(),
        }

    def metadata_process_sample(self, raw_metadata: Dict[str, Any], sample: Dict[str, Any]) -> None:
        self.gnn_model.metadata_process_sample(raw_metadata["gnn_metadata"], sample["graph"])

    def metadata_finalize(self, raw_metadata: Dict[str, Any]) -> None:
        self.gnn_model.metadata_finalize(raw_metadata["gnn_metadata"])

    def batch_init(self) -> Dict[str, Any]:
        return {
            "gnn_batch": self.gnn_model.batch_init(),
            # Localization information
            "rewritable_loc_node_idxs": [],
            "rewritable_loc_to_sample_id": [],
            "sample_has_bug": [],
            "sample_to_correct_loc_idx": [],
            # Text rewrites
            "text_rewrite_node_idxs": [],
            "text_rewrite_replacement_ids": [],
            "text_rewrite_to_loc_group": [],
            "text_rewrite_correct_idxs": [],
            # Var Misuse
            "varswap_node_idxs": [],
            "varswap_replacement_node_idxs": [],
            "varswap_to_loc_group": [],
            "varswap_correct_idxs": [],
            # Arg Swaps
            "argswap_node_idxs": [],
            "argswap_swapped_node_pair_idxs": [],
            "argswap_to_loc_group": [],
            "argswap_correct_idxs": [],
        }

    def _compute_rewrite_data(
        self, sample: Dict[str, Any], rewritable_loc_node_idxs: List[int]
    ) -> Tuple[
        RewriteModuleSelectionInformation,
        RewriteModuleSelectionInformation,
        RewriteModuleSelectionInformation,
    ]:
        # TODO: This function is awful. Any way to improve it?

        target_fix_action_idx = sample["target_fix_action_idx"]
        if target_fix_action_idx is not None:
            target_node_idx = sample["graph"]["reference_nodes"][target_fix_action_idx]
        else:
            target_node_idx = None

        call_args = defaultdict(list)
        all_nodes = sample["graph"]["nodes"]
        for child_edge in sample["graph"]["edges"]["Child"]:
            if len(child_edge) == 3:
                parent_idx, child_idx, edge_type = child_edge
            else:
                parent_idx, child_idx = child_edge
                edge_type = None
            if edge_type == "args" and all_nodes[parent_idx] == "Call":
                call_args[parent_idx].append(child_idx)

        incorrect_node_id_to_text_rewrite_target: Dict[int, List[int]] = defaultdict(list)
        text_rewrite_original_idx: Dict[int, List[int]] = defaultdict(list)
        correct_text_rewrite_target: Optional[Tuple[int, int]] = None

        misused_node_ids_to_candidates_symbol_node_ids: Dict[int, List[int]] = defaultdict(list)
        varmisuse_rewrite_original_idx: Dict[int, List[int]] = defaultdict(list)
        correct_misused_node_and_candidate: Optional[Tuple[int, int]] = None

        call_node_ids_to_candidate_swapped_node_ids: Dict[int, List[jnp.ndarray]] = defaultdict(list)
        swapped_rewrite_original_ids: Dict[int, List[int]] = defaultdict(list)
        correct_swapped_call_and_pair: Optional[Tuple[int, int]] = None

        for i, (node_idx, (_, rewrite_data), (rewrite_scout, rewrite_metadata),) in enumerate(
            zip(
                sample["graph"]["reference_nodes"],
                sample["candidate_rewrites"],
                sample["candidate_rewrite_metadata"],
            )
        ):
            # TODO: this should be a switch, to optimize training, when we only need to look at the correct rewrite...
            if False and node_idx != target_node_idx:
                # During training we only care about the target rewrite.
                continue
            is_target_action = target_fix_action_idx == i

            if rewrite_scout == "VariableMisuseRewriteScout":
                if is_target_action:
                    correct_misused_node_and_candidate = (
                        node_idx,
                        len(misused_node_ids_to_candidates_symbol_node_ids[node_idx]),
                    )
                misused_node_ids_to_candidates_symbol_node_ids[node_idx].append(rewrite_metadata)
                varmisuse_rewrite_original_idx[node_idx].append(i)

            elif rewrite_scout == "ArgSwapRewriteScout":
                arg_node_idxs = call_args[node_idx]
                swapped_arg_node_idxs = jnp.array(
                    (arg_node_idxs[rewrite_data[0]], arg_node_idxs[rewrite_data[1]])
                )  # The to-be swapped node idxs. Stored as an array for easier shifting during batching.
                if is_target_action:
                    correct_swapped_call_and_pair = (
                        node_idx,
                        len(call_node_ids_to_candidate_swapped_node_ids[node_idx]),
                    )
                call_node_ids_to_candidate_swapped_node_ids[node_idx].append(swapped_arg_node_idxs)
                swapped_rewrite_original_ids[node_idx].append(i)
            else:
                if is_target_action:
                    correct_text_rewrite_target = (
                        node_idx,
                        len(incorrect_node_id_to_text_rewrite_target[target_node_idx]),
                    )

                incorrect_node_id_to_text_rewrite_target[node_idx].append(
                    self.rewrite_vocabulary.get_id_or_unk(rewrite_data)
                )
                text_rewrite_original_idx[node_idx].append(i)
        repr_location_group_ids = {location_node_idx: i for i, location_node_idx in enumerate(rewritable_loc_node_idxs)}

        def to_flat_node_selection(
            candidates: Dict[int, List],
            correct_candidate: Optional[Tuple[int, int]],
            original_rewrite_idxs: Dict[int, List[int]],
        ) -> RewriteModuleSelectionInformation:
            (
                node_selection_repr_node_ids,
                candidate_node_ids,
                candidate_node_to_repr_node,
            ) = ([], [], [])
            flat_original_rewrite_idxs: List[int] = []

            correct_idx = None
            for repr_node_id, candidate_nodes in candidates.items():
                if correct_candidate is not None and correct_candidate[0] == repr_node_id:
                    correct_idx = len(candidate_node_ids) + correct_candidate[1]
                node_selection_repr_node_ids.extend(repr_node_id for _ in candidate_nodes)
                candidate_node_ids.extend(candidate_nodes)
                group_idx = repr_location_group_ids[repr_node_id]
                flat_original_rewrite_idxs.extend(original_rewrite_idxs[repr_node_id])

                candidate_node_to_repr_node.extend((group_idx for _ in candidate_nodes))

            return RewriteModuleSelectionInformation(
                node_idxs=node_selection_repr_node_ids,
                replacement_ids=candidate_node_ids,
                loc_groups=candidate_node_to_repr_node,
                correct_choice_idx=correct_idx,
                original_idxs=flat_original_rewrite_idxs,
            )

        results = (
            to_flat_node_selection(
                incorrect_node_id_to_text_rewrite_target,
                correct_text_rewrite_target,
                text_rewrite_original_idx,
            ),
            to_flat_node_selection(
                misused_node_ids_to_candidates_symbol_node_ids,
                correct_misused_node_and_candidate,
                varmisuse_rewrite_original_idx,
            ),
            to_flat_node_selection(
                call_node_ids_to_candidate_swapped_node_ids,
                correct_swapped_call_and_pair,
                swapped_rewrite_original_ids,
            ),
        )

        return results

    def batch_can_fit(self, raw_batch: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        return self.gnn_model.batch_can_fit(raw_batch["gnn_batch"], sample["graph"])

    def batch_add_sample(self, raw_batch: Dict[str, Any], sample: Dict[str, Any]) -> None:
        # JSON has format
        # {
        #   "graph": {... stuff for the GNN, but includes a list of reference nodes},
        #   "candidate_rewrites": [... list of [rewrite_type, rewrite_data] pairs ...],
        #   "candidate_rewrite_metadata": [... list of [rewrite_scout, rewrite_metadata] pairs ...],
        #   "target_fix_action_idx": optional INDEX into graph["reference_nodes"], indicating the buggy node.
        #                            None indicates the sample is non-buggy.
        #   "package_name": NAME,
        #   "package_version": VERSION,
        # }
        # Pull out the current number of nodes before updating the graph info, as we'll need to
        # reindex some references in the rewriter choosing-specific bits:
        sample_id = len(raw_batch["sample_has_bug"])
        node_idx_batch_offset = raw_batch["gnn_batch"]["num_nodes"]
        self.gnn_model.batch_add_sample(raw_batch["gnn_batch"], sample["graph"])

        # --- Localization information
        loc_idx_batch_offset = len(raw_batch["rewritable_loc_node_idxs"])
        correct_loc_idx = sample["target_fix_action_idx"]

        # The list of reference nodes contains duplicates, so we strip this down here
        rewritable_loc_node_idxs, inv = jnp.unique(jnp.array(sample["graph"]["reference_nodes"]), return_inverse=True)

        if correct_loc_idx is None:
            raw_batch["sample_has_bug"].append(False)
            raw_batch["sample_to_correct_loc_idx"].append(-10000)
        else:
            raw_batch["sample_has_bug"].append(True)
            # We need to reflect the uniquefication above here:
            correct_loc_idx = inv[correct_loc_idx]
            raw_batch["sample_to_correct_loc_idx"].append(correct_loc_idx + loc_idx_batch_offset)

        for loc_node_idx in rewritable_loc_node_idxs:
            raw_batch["rewritable_loc_node_idxs"].append(loc_node_idx + node_idx_batch_offset)
            raw_batch["rewritable_loc_to_sample_id"].append(sample_id)

        # --- Some common footwork that disentangles the different rewrites into more
        (
            tr_node_sel_info,
            vs_node_sel_info,
            as_node_sel_info,
        ) = self._compute_rewrite_data(sample, rewritable_loc_node_idxs.tolist())

        # Text rewrites:
        if tr_node_sel_info.correct_choice_idx is not None:
            # This is an index into the list of text rewrites in this batch, so we need to shift it appropriately:
            raw_batch["text_rewrite_correct_idxs"].append(
                tr_node_sel_info.correct_choice_idx + len(raw_batch["text_rewrite_to_loc_group"])
            )
        raw_batch["text_rewrite_node_idxs"].extend(
            [node_id + node_idx_batch_offset for node_id in tr_node_sel_info.node_idxs]
        )
        # The replacement IDs refer to a vocabulary, so need no shifting:
        raw_batch["text_rewrite_replacement_ids"].extend(tr_node_sel_info.replacement_ids)
        raw_batch["text_rewrite_to_loc_group"].extend(
            [loc_id + loc_idx_batch_offset for loc_id in tr_node_sel_info.loc_groups]
        )

        # Var swapping:
        if vs_node_sel_info.correct_choice_idx is not None:
            # This is an index into the list of var swaps in this batch, so we need to shift it appropriately:
            raw_batch["varswap_correct_idxs"].append(
                vs_node_sel_info.correct_choice_idx + len(raw_batch["varswap_to_loc_group"])
            )
        raw_batch["varswap_node_idxs"].extend(
            [node_id + node_idx_batch_offset for node_id in vs_node_sel_info.node_idxs]
        )
        # The replacement IDs refer to nodes, so need shifting:
        raw_batch["varswap_replacement_node_idxs"].extend(
            [replacement_node_id + node_idx_batch_offset for replacement_node_id in vs_node_sel_info.replacement_ids]
        )
        raw_batch["varswap_to_loc_group"].extend(
            [loc_id + loc_idx_batch_offset for loc_id in vs_node_sel_info.loc_groups]
        )

        # Arg swapping:
        if as_node_sel_info.correct_choice_idx is not None:
            # This is an index into the list of arg swaps in this batch, so we need to shift it appropriately:
            raw_batch["argswap_correct_idxs"].append(
                as_node_sel_info.correct_choice_idx + len(raw_batch["argswap_to_loc_group"])
            )
        raw_batch["argswap_node_idxs"].extend(
            [node_id + node_idx_batch_offset for node_id in as_node_sel_info.node_idxs]
        )
        # The replacement IDs refer to pairs of nodes, so need shifting:
        raw_batch["argswap_swapped_node_pair_idxs"].extend(
            [replacement_node_id + node_idx_batch_offset for replacement_node_id in as_node_sel_info.replacement_ids]
        )
        raw_batch["argswap_to_loc_group"].extend(
            [loc_id + loc_idx_batch_offset for loc_id in as_node_sel_info.loc_groups]
        )

    def batch_pad(self, raw_batch: Dict[str, Any]) -> None:
        self.gnn_model.batch_pad(raw_batch["gnn_batch"])

        # --- First, get all the target sizes that we want to pad up to:
        num_real_samples = len(raw_batch["sample_has_bug"])
        # We need at least one extra sample for the remaining padding to work:
        target_num_samples = self._get_padded_size("num_samples", num_real_samples + 1)
        num_pad_samples = target_num_samples - num_real_samples

        num_real_locs = len(raw_batch["rewritable_loc_node_idxs"])
        # We need at least one extra location for the remaining padding to work:
        target_num_locs = self._get_padded_size("num_locs", num_real_locs + 1)
        num_pad_locs = target_num_locs - num_real_locs

        num_real_tr = len(raw_batch["text_rewrite_node_idxs"])
        target_num_tr = self._get_padded_size("num_text_rewrites", num_real_tr)
        num_pad_tr = target_num_tr - num_real_tr
        num_real_correct_tr = len(raw_batch["text_rewrite_correct_idxs"])
        target_num_correct_tr = self._get_padded_size("num_correct_text_rewrites", num_real_correct_tr)
        num_pad_correct_tr = target_num_correct_tr - num_real_correct_tr

        num_real_vs = len(raw_batch["varswap_node_idxs"])
        target_num_vs = self._get_padded_size("num_varswaps", num_real_vs)
        num_pad_vs = target_num_vs - num_real_vs
        num_real_correct_vs = len(raw_batch["varswap_correct_idxs"])
        target_num_correct_vs = self._get_padded_size("num_correct_varswaps", num_real_correct_vs)
        num_pad_correct_vs = target_num_correct_vs - num_real_correct_vs

        num_real_as = len(raw_batch["argswap_node_idxs"])
        target_num_as = self._get_padded_size("num_argswaps", num_real_as)
        num_pad_as = target_num_as - num_real_as
        num_real_correct_as = len(raw_batch["argswap_correct_idxs"])
        target_num_correct_as = self._get_padded_size("num_correct_argswaps", num_real_correct_as)
        num_pad_correct_as = target_num_correct_as - num_real_correct_as

        # --- Now actually add the padding:
        pad_node_id = raw_batch["gnn_batch"]["num_nodes"] - 1
        pad_sample_id = target_num_samples - 1
        pad_loc_id = target_num_locs - 1

        raw_batch["sample_has_bug"].extend([False] * num_pad_samples)
        raw_batch["sample_to_correct_loc_idx"].extend([-10000] * num_pad_samples)
        raw_batch["sample_is_nonpad"] = [True] * num_real_samples + [False] * num_pad_samples

        raw_batch["rewritable_loc_node_idxs"].extend([pad_node_id] * num_pad_locs)
        raw_batch["rewritable_loc_to_sample_id"].extend([pad_sample_id] * num_pad_locs)

        raw_batch["text_rewrite_node_idxs"].extend([pad_node_id] * num_pad_tr)
        raw_batch["text_rewrite_replacement_ids"].extend([0] * num_pad_tr)
        raw_batch["text_rewrite_to_loc_group"].extend([pad_loc_id] * num_pad_tr)
        raw_batch["text_rewrite_correct_idxs"].extend([target_num_tr - 1] * num_pad_correct_tr)
        raw_batch["text_rewrite_correct_is_nonpad"] = [True] * num_real_correct_tr + [False] * num_pad_correct_tr

        raw_batch["varswap_node_idxs"].extend([pad_node_id] * num_pad_vs)
        raw_batch["varswap_replacement_node_idxs"].extend([pad_node_id] * num_pad_vs)
        raw_batch["varswap_to_loc_group"].extend([pad_loc_id] * num_pad_vs)
        raw_batch["varswap_correct_idxs"].extend([target_num_vs - 1] * num_pad_correct_vs)
        raw_batch["varswap_correct_is_nonpad"] = [True] * num_real_correct_vs + [False] * num_pad_correct_vs

        raw_batch["argswap_node_idxs"].extend([pad_node_id] * num_pad_as)
        raw_batch["argswap_swapped_node_pair_idxs"].extend([jnp.array((pad_node_id, pad_node_id))] * num_pad_as)
        raw_batch["argswap_to_loc_group"].extend([pad_loc_id] * num_pad_as)
        raw_batch["argswap_correct_idxs"].extend([target_num_as - 1] * num_pad_correct_as)
        raw_batch["argswap_correct_is_nonpad"] = [True] * num_real_correct_as + [False] * num_pad_correct_as

    def batch_finalize(self, raw_batch: Dict[str, Any]) -> Tuple[GNNBuxlabBatchFeatures, RewriteChooserBatchLabels]:
        features = GNNBuxlabBatchFeatures(
            sample_is_nonpad=jnp.array(raw_batch["sample_is_nonpad"]),
            # GNN information:
            graph_data=self.gnn_model.batch_finalize(raw_batch["gnn_batch"]),
            # Localization information
            rewritable_loc_node_idxs=jnp.array(raw_batch["rewritable_loc_node_idxs"]),
            rewritable_loc_to_sample_id=jnp.array(raw_batch["rewritable_loc_to_sample_id"]),
            # Text rewrites
            text_rewrite_node_idxs=jnp.array(raw_batch["text_rewrite_node_idxs"]),
            text_rewrite_replacement_ids=jnp.array(raw_batch["text_rewrite_replacement_ids"]),
            text_rewrite_to_loc_group=jnp.array(raw_batch["text_rewrite_to_loc_group"]),
            # Var Misuse
            varswap_node_idxs=jnp.array(raw_batch["varswap_node_idxs"]),
            varswap_replacement_node_idxs=jnp.array(raw_batch["varswap_replacement_node_idxs"]),
            varswap_to_loc_group=jnp.array(raw_batch["varswap_to_loc_group"]),
            # Arg swaps
            argswap_node_idxs=jnp.array(raw_batch["argswap_node_idxs"]),
            argswap_swapped_node_pair_idxs=jnp.array(raw_batch["argswap_swapped_node_pair_idxs"]),
            argswap_to_loc_group=jnp.array(raw_batch["argswap_to_loc_group"]),
        )
        labels = RewriteChooserBatchLabels(
            sample_has_bug=jnp.array(raw_batch["sample_has_bug"]),
            sample_to_correct_loc_idx=jnp.array(raw_batch["sample_to_correct_loc_idx"]),
            # Text rewrites
            text_rewrite_correct_idxs=jnp.array(raw_batch["text_rewrite_correct_idxs"]),
            text_rewrite_correct_is_nonpad=jnp.array(raw_batch["text_rewrite_correct_is_nonpad"]),
            # Var Misuse
            varswap_correct_idxs=jnp.array(raw_batch["varswap_correct_idxs"]),
            varswap_correct_is_nonpad=jnp.array(raw_batch["varswap_correct_is_nonpad"]),
            # Arg swaps
            argswap_correct_idxs=jnp.array(raw_batch["argswap_correct_idxs"]),
            argswap_correct_is_nonpad=jnp.array(raw_batch["argswap_correct_is_nonpad"]),
        )

        return features, labels

    def build_module(self) -> GNNBuxlabModule:
        return GNNBuxlabModule(
            rewrite_vocab_size=len(self.rewrite_vocabulary),
            localization_module_type=self.localization_module_type,
            buggy_samples_weight_schedule=self.buggy_samples_weight_schedule,
            repair_weight_schedule=self.repair_weight_schedule,
            gnn_module_maker=self.gnn_model.build_module,
            use_all_gnn_layer_outputs=self.use_all_gnn_layer_outputs,
        )
