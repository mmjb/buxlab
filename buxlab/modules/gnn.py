from typing import Generic, Optional, Callable, List, Dict, Any, Tuple
from typing_extensions import Literal
from dataclasses import dataclass
import math
import re

from flax.struct import PyTreeNode
import flax.linen as nn
import gin
import jax.numpy as jnp

from ..utils.buxlab_interface import AbstractBuxlabModel
from ..utils.vocabulary import split_identifier_into_parts
from .string_embedding_modules import (
    SubtokenEmbedderModel,
    TokenListEmbedderModel,
    BatchedTokenListType,
    TokenListEmbedderModuleType,
)

from .mlp import MLP

SMALL_NUMBER = 1e-7

IS_IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
OPEN_VOCAB_EDGE_TYPE = "HasSubtoken"


def add_open_vocab_nodes_and_edges(graph: Dict[str, Any]) -> None:
    token_nodes = set()
    if "NextToken" not in graph["edges"]:
        return
    for n1, n2 in graph["edges"]["NextToken"]:
        token_nodes.add(n1)
        token_nodes.add(n2)

    vocab_nodes: Dict[str, int] = {}
    vocab_edges: List[Tuple[int, int]] = []

    all_nodes = graph["nodes"]
    for node_idx in token_nodes:
        token_str = all_nodes[node_idx]
        if not IS_IDENTIFIER.match(token_str):
            continue
        for subtoken in split_identifier_into_parts(token_str):
            subtoken_node_idx = vocab_nodes.get(subtoken)
            if subtoken_node_idx is None:
                subtoken_node_idx = len(all_nodes)
                all_nodes.append(subtoken)
                vocab_nodes[subtoken] = subtoken_node_idx
            vocab_edges.append((node_idx, subtoken_node_idx))

    graph["edges"][OPEN_VOCAB_EDGE_TYPE] = vocab_edges


@gin.configurable
class RelationalMP(nn.Module):
    """Relational message passing, using different message functions for each relation/edge
    type."""

    num_edge_types: int
    hidden_dim: int
    message_function_depth: int

    def setup(self):
        super().setup()

        self.edge_mlps = [
            MLP(
                dims=[self.edge_message_size] * (self.message_function_depth + 1),
                use_bias=False,
            )
            for _ in range(self.num_edge_types)
        ]

    @property
    def edge_message_size(self) -> int:
        return self.hidden_dim

    @property
    def state_update_size(self) -> int:
        return self.hidden_dim

    def __call__(self, node_states: jnp.ndarray, adj_lists: List[jnp.ndarray]):
        all_msg_list: List[jnp.ndarray] = []  # all messages exchanged between nodes
        all_tgts_list: List[jnp.ndarray] = []  # [E] - list of targets for all messages

        for edge_type, adj_list in enumerate(adj_lists):
            srcs = adj_list[:, 0]
            tgts = adj_list[:, 1]

            messages = self.edge_mlps[edge_type](jnp.concatenate((node_states[srcs], node_states[tgts]), axis=1))
            messages = nn.relu(messages)

            all_msg_list.append(messages)
            all_tgts_list.append(tgts)

        all_messages: jnp.ndarray = jnp.concatenate(all_msg_list, axis=0)  # type: ignore # [E, edge_message_size]
        all_targets: jnp.ndarray = jnp.concatenate(all_tgts_list, axis=0)  # type: ignore # [E]

        return self._aggregate_messages(all_messages, all_targets, num_nodes=node_states.shape[0])

    def _aggregate_messages(
        self,
        messages: jnp.ndarray,
        targets: jnp.ndarray,
        num_nodes: int,
    ):
        aggregated_messages = jnp.zeros(shape=(num_nodes, messages.shape[-1]), dtype=messages.dtype)
        return aggregated_messages.at[targets].add(messages)


@gin.configurable
class RelationalMultiAggrMP(RelationalMP):
    """Relational message passing, but using four different aggregation strategies (sum, mean, stdev, max)."""

    msg_dim: int

    @property
    def edge_message_size(self) -> int:
        return 3 * self.msg_dim  # Aggregated as sum, mean, stdev (of mean messages), max

    @property
    def state_update_size(self) -> int:
        return 4 * self.msg_dim

    def _aggregate_messages(self, messages: jnp.ndarray, targets: jnp.ndarray, num_nodes: int):
        sum_messages, mean_messages, max_messages = jnp.split(messages, indices_or_sections=4, axis=1)

        sum_aggregated_messages = jnp.zeros(shape=(num_nodes, sum_messages.shape[-1]), dtype=messages.dtype)
        sum_aggregated_messages = sum_aggregated_messages.at[targets].add(sum_messages)

        max_aggregated_messages = jnp.full(
            shape=(num_nodes, max_messages.shape[-1]),
            fill_value=jnp.iinfo(messages.dtype).min,
        )
        max_aggregated_messages = max_aggregated_messages.at[targets].max(max_messages)

        mean_aggregated_messages = jnp.zeros(shape=(num_nodes, mean_messages.shape[-1]), dtype=messages.dtype)
        num_mean_messages = jnp.zeros(shape=(num_nodes, 1), dtype=messages.dtype)
        num_mean_messages = num_mean_messages.at[targets].add(
            jnp.ones(shape=(mean_messages.shape[0], 1), dtype=messages.dtype)
        )
        mean_aggregated_messages = mean_aggregated_messages.at[targets].add(mean_messages)
        mean_aggregated_messages = mean_aggregated_messages / num_mean_messages

        std_aggregated_messages = jnp.zeros(shape=(num_nodes, mean_messages.shape[-1]), dtype=messages.dtype)
        per_node_message_stdev = nn.relu(mean_messages.pow(2) - mean_aggregated_messages[targets].pow(2)) + SMALL_NUMBER
        std_aggregated_messages = jnp.sqrt(std_aggregated_messages.at[targets].add(per_node_message_stdev))

        state_updates = jnp.concatenate(
            (
                sum_aggregated_messages,
                mean_aggregated_messages,
                std_aggregated_messages,
                max_aggregated_messages,
            ),
            axis=1,
        )

        return state_updates


@gin.configurable
class BOOMLayer(nn.Module):
    """Shallow MLP with large intermediate layer.
    Named in Sect. 3 of https://arxiv.org/pdf/1911.11423.pdf:
    'Why Boom? We take a vector from small (1024) to big (4096) to small (1024). It’s really not
     that hard to visualize - use your hands if you need to whilst shouting "boooOOOOmmm".'
    """

    inout_dim: int
    intermediate_dim: int
    dropout_rate: float = 0.1

    def setup(self):
        self.linear1 = nn.Dense(features=self.intermediate_dim, use_bias=False)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.linear2 = nn.Dense(features=self.inout_dim, use_bias=False)

    def __call__(self, x, train: bool = False):
        return self.linear2(self.dropout(nn.leaky_relu(self.linear1(x)), deterministic=not train))


@gin.configurable
class GNNBlock(nn.Module):
    """Block in a GNN, following a Transformer-like residual structure, using the "Pre-Norm" style
    and ReZero weighting using \alpha:
      v' = v + \alpha * Dropout(NeighbourHoodAtt(LN(v))))
      v = v' + \alpha * Linear(Dropout(Act(Linear(LN(v'))))))

    Pre-Norm reference: https://arxiv.org/pdf/2002.04745v1.pdf
    ReZero reference: https://arxiv.org/pdf/2003.04887v1.pdf
    ReZero' (with \alpha a vector instead of scalar): https://arxiv.org/pdf/2103.17239.pdf
    """

    type: Literal["MultiAggr", "Plain"] = "Plain"
    msg_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    hidden_dim: int = 128
    num_edge_types: int = 1
    num_towers: int = 1
    intermediate_dim: int = 512
    message_function_depth: int = 0
    dropout_rate: float = 0.0
    use_rezero_scaling: bool = True

    def setup(self):
        if self.use_rezero_scaling:
            self.alpha = self.param(
                name="rezero_alpha",
                init_fn=lambda _: jnp.full(shape=(1,), fill_value=SMALL_NUMBER),
            )

        # We implement the "towers" trick introduced by Gilmer et al., in which several message passing mechanisms work
        # in parallel on subsets. We slice the overall node representations into a part for each of these:
        assert self.hidden_dim % self.num_towers == 0, "Number of heads needs to divide GNN hidden dim."
        mp_layer_in_dim = self.hidden_dim // self.num_towers
        mp_layers = []
        for _ in range(self.num_towers):
            if self.type.lower() == "MultiAggr".lower():
                mp_layers.append(
                    RelationalMultiAggrMP(
                        hidden_dim=mp_layer_in_dim,
                        num_edge_types=self.num_edge_types,
                        message_function_depth=self.message_function_depth,
                        msg_dim=mp_layer_in_dim,
                    )
                )
            elif self.type.lower() == "Plain".lower():
                mp_layers.append(
                    RelationalMP(
                        hidden_dim=mp_layer_in_dim,
                        num_edge_types=self.num_edge_types,
                        message_function_depth=self.message_function_depth,
                    )
                )
            else:
                raise ValueError(f"Unknown GNN type {self.type}.")
        self.mp_layers = mp_layers

        self.msg_out_projection = nn.Dense(features=self.hidden_dim, use_bias=False)

        self.mp_norm_layer = nn.LayerNorm()

        if self.intermediate_dim > 0:
            self.boom_layer: Optional[BOOMLayer] = BOOMLayer(
                inout_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                dropout_rate=self.dropout_rate,
            )
            self.boom_norm_layer: Optional[nn.Module] = nn.LayerNorm()
        else:
            self.boom_layer = None
            self.boom_norm_layer = None

        # We will use this one dropout layer everywhere, as it's stateless:
        self.dropout_layer = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, node_representations, adj_lists, train: bool = False):
        """
        Args:
            node_representations: float tensor of shape (num_nodes, self.hidden_dim)
            adj_lists: list of (num_edges, 2) tensors (one per edge-type)
        Returns:
            node_representations: float (num_graphs, self.hidden_dim) tensor
        """
        aggregated_messages = []
        for i, sliced_node_reprs in enumerate(
            jnp.split(node_representations, indices_or_sections=self.num_towers, axis=1)
        ):
            aggregated_messages.append(self.mp_layers[i](sliced_node_reprs, adj_lists))

        new_representations = self.msg_out_projection(jnp.concatenate(aggregated_messages, axis=-1))
        new_representations = self.msg_activation(new_representations)
        new_representations = self.dropout_layer(new_representations, deterministic=not train)
        if self.use_rezero_scaling:
            new_representations = self.alpha * new_representations
        node_representations = node_representations + new_representations

        if self.boom_layer is not None and self.boom_norm_layer is not None:
            boomed_representations = self.dropout_layer(
                self.boom_layer(self.boom_norm_layer(node_representations), train=train),
                deterministic=not train,
            )
            if self.use_rezero_scaling:
                boomed_representations = self.alpha * boomed_representations
            node_representations = node_representations + boomed_representations

        return node_representations


class GNNBatch(PyTreeNode, Generic[BatchedTokenListType]):
    """General data structure for holding information about graphs in a batch.
    Note that batches of unequally sized graphs are formed from multiple graphs by
    combining into one large disconnected graph, where each new potential addition is tested
    to check that it will not exceed the allowed number of nodes or edges per batch.

    Args:
        num_graphs: total number of graphs in the batch.
        num_nodes: total number of nodes in the batch, V. Should be limited to a maximum.
        node_labels: as determined by the used used StringEmbeddingModule.
        adjacency_lists: list of all edges in the batch, one tensor per each edge type.
            list, len num_edge_types, elements [num edges, 2] int tensors
        node_to_graph: vector of indices of length V. Mapping from nodes to the graphs
            to which they belong.
    """

    num_graphs: int
    num_nodes: int
    node_labels: BatchedTokenListType
    adjacency_lists: List[jnp.ndarray]  # list of length num_edge_types, [num edges, 2] int tensors
    node_to_graph: jnp.ndarray  # [V] long


class GNNModule(nn.Module, Generic[TokenListEmbedderModuleType]):
    num_edge_types: int
    node_label_embedder_maker: Callable[[], TokenListEmbedderModuleType]
    hidden_dim: int
    num_layers: int
    add_backwards_edges: bool

    def setup(self):
        self.node_label_embedder = self.node_label_embedder_maker()

        num_effective_edge_types = self.num_edge_types
        if self.add_backwards_edges:
            num_effective_edge_types *= 2

        self.gnn_blocks = [
            GNNBlock(hidden_dim=self.hidden_dim, num_edge_types=num_effective_edge_types)
            for _ in range(self.num_layers)
        ]

    def __call__(self, input: GNNBatch, train: bool = False) -> List[jnp.ndarray]:
        """
        Args:
            input: GNNBatch object describing a batch of graphs.

        Returns:
            all_node_representations: list of float32 (num_graphs, self.hidden_dim) tensors,
                one for the result of each timestep of the GNN (and the initial one)
        """
        # We may need to introduce additional edges:
        adj_lists = list(input.adjacency_lists)
        if self.add_backwards_edges:
            adj_lists.extend([jnp.flip(adj_list, axis=(1,)) for adj_list in input.adjacency_lists])

        # Delegate embedding of the node labels to the submodule:
        cur_node_representations = self.node_label_embedder(input.node_labels)

        # Actually do message passing:
        all_node_representations: List[jnp.ndarray] = [cur_node_representations]
        for gnn_block in self.gnn_blocks:
            cur_node_representations = gnn_block(
                node_representations=cur_node_representations,
                adj_lists=adj_lists,
                train=train,
            )
            all_node_representations.append(cur_node_representations)

        return all_node_representations


@gin.configurable
@dataclass
class GNNModel(
    AbstractBuxlabModel[Dict[str, Any], GNNBatch, GNNModule],
    Generic[BatchedTokenListType, TokenListEmbedderModuleType],
):
    node_label_model: TokenListEmbedderModel[
        BatchedTokenListType, TokenListEmbedderModuleType
    ] = SubtokenEmbedderModel()
    hidden_dim: int = 128
    num_layers: int = 8
    add_backwards_edges: bool = True
    use_open_vocab_graph_ext: bool = True
    batch_max_num_graphs: Optional[int] = None
    batch_max_num_nodes: Optional[int] = 8000
    batch_max_num_edges: Optional[int] = 250000

    def metadata_init(self) -> Dict[str, Any]:
        metadata = {
            "edge_set": set(),
            "node_labels": self.node_label_model.metadata_init(),
        }

        if self.use_open_vocab_graph_ext:
            metadata["edge_set"].add(OPEN_VOCAB_EDGE_TYPE)

        return metadata

    def metadata_process_sample(self, raw_metadata: Dict[str, Any], sample: Dict[str, Any]) -> None:
        raw_metadata["edge_set"].update(sample["edges"].keys())
        for node_label in sample["nodes"]:
            self.node_label_model.metadata_process_sample(raw_metadata["node_labels"], node_label)

    def metadata_finalize(self, raw_metadata: Dict[str, Any]) -> None:
        self.edge_vocabulary: List[str] = list(raw_metadata["edge_set"])
        self.node_label_model.metadata_finalize(raw_metadata["node_labels"])

    def batch_init(self) -> Dict[str, Any]:
        return {
            "num_graphs": 0,
            "num_nodes": 0,
            "num_edges": 0,
            "adjacency_lists": {et: [] for et in self.edge_vocabulary},
            "node_to_graph": [],
            "node_labels": self.node_label_model.batch_init(),
        }

    def batch_can_fit(self, raw_batch: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        if self.use_open_vocab_graph_ext:
            # this is a bit ugly because it changes the underlying sample...
            add_open_vocab_nodes_and_edges(sample)
        sample_num_nodes = len(sample["nodes"])
        sample_num_edges = sum(len(adj_list) for adj_list in sample["edges"].values())

        # Decide if this batch is full:
        return (
            (raw_batch["num_graphs"] + 1 < (self.batch_max_num_graphs or math.inf))
            and (raw_batch["num_nodes"] + sample_num_nodes < (self.batch_max_num_nodes or math.inf))
            and (raw_batch["num_edges"] + sample_num_edges < (self.batch_max_num_edges or math.inf))
        )

    def batch_add_sample(self, raw_batch: Dict[str, Any], sample: Dict[str, Any]) -> None:
        # JSON has format
        # {
        #   "nodes": [...list of strings...],
        #   "edges": {edge_type_name: [...list of [source node id, target node id, optional label] tuples]},
        #   "reference_nodes": [...list of node ids...],
        #   "text": SOURCE_AS_TEXT,
        #   "path": SOURCE_FILE_PATH,
        #   "code_range": [[START_LINE, START_COL], [END_LINE, END_COL]]
        # }
        sample_num_nodes = len(sample["nodes"])
        sample_num_edges = sum(len(adj_list) for adj_list in sample["edges"].values())
        sample_id_in_batch = raw_batch["num_graphs"]

        # Collect the actual graph information:
        for edge_type, adj_list in sample["edges"].items():
            # adj_list is a list of pairs and triples (if we have edge information). Clean up:
            adj_list = jnp.array([(e[0], e[1]) for e in adj_list], dtype=jnp.int32)
            if adj_list.shape[0] == 0:
                adj_list = jnp.zeros((0, 2), dtype=jnp.int32)
            shifted_adj_list = adj_list + raw_batch["num_nodes"]
            raw_batch["adjacency_lists"][edge_type].append(shifted_adj_list)
        raw_batch["node_to_graph"].append(jnp.full(shape=(sample_num_nodes,), fill_value=sample_id_in_batch))
        for node_label in sample["nodes"]:
            self.node_label_model.batch_add_sample(raw_batch["node_labels"], node_label)

        # Some housekeeping information:
        raw_batch["num_graphs"] += 1
        raw_batch["num_nodes"] += sample_num_nodes
        raw_batch["num_edges"] += sample_num_edges

    def batch_pad(self, raw_batch: Dict[str, Any]) -> None:
        num_real_nodes = raw_batch["num_nodes"]

        # We require an extra node for edge padding, so add that to the size:
        target_num_nodes = self._get_padded_size("nodes", num_real_nodes + 1)
        num_pad_nodes = target_num_nodes - num_real_nodes
        raw_batch["num_nodes"] = target_num_nodes
        raw_batch["num_pad_nodes"] = num_pad_nodes
        raw_batch["num_graphs"] += 1
        raw_batch["node_to_graph"].append(jnp.full(shape=(num_pad_nodes,), fill_value=raw_batch["num_graphs"]))
        self.node_label_model.batch_add_padding(raw_batch["node_labels"], target_num_samples=target_num_nodes)

        for et in self.edge_vocabulary:
            adj_lists = raw_batch["adjacency_lists"].get(et, [])
            num_real_edges = sum(len(a) for a in adj_lists)
            target_num_edges = self._get_padded_size(f"adjacency_list:{et}", num_real_edges)
            num_pad_edges = target_num_edges - num_real_edges
            if num_pad_edges > 0:
                adj_lists.append(
                    jnp.full(
                        shape=(num_pad_edges, 2),
                        fill_value=num_real_nodes,  # This happens to be the ID of the first padding node
                        dtype=jnp.int32,
                    )
                )

    def batch_finalize(self, raw_batch: Dict[str, Any]) -> GNNBatch:
        adjacency_lists = []
        for et in self.edge_vocabulary:
            adj_lists = raw_batch["adjacency_lists"].get(et, [])
            if len(adj_lists) > 0:
                adjacency_lists.append(jnp.concatenate(adj_lists, axis=0))
            else:
                adjacency_lists.append(jnp.zeros(shape=(0, 2), dtype=jnp.int32))

        return GNNBatch(
            num_graphs=raw_batch["num_graphs"],
            num_nodes=raw_batch["num_nodes"],
            node_labels=self.node_label_model.batch_finalize(raw_batch["node_labels"]),
            adjacency_lists=adjacency_lists,
            node_to_graph=jnp.concatenate(raw_batch["node_to_graph"], axis=0),
        )

    def build_module(self) -> GNNModule:
        return GNNModule(
            num_edge_types=len(self.edge_vocabulary),
            node_label_embedder_maker=self.node_label_model.build_module,
            add_backwards_edges=self.add_backwards_edges,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
