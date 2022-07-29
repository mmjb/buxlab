from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Dict, Any, Iterable, TypeVar, Generic, List
from collections import Counter

import flax.linen as nn
import gin
import jax.numpy as jnp
from flax.struct import PyTreeNode

from ..utils.buxlab_interface import AbstractBuxlabModel
from ..utils.vocabulary import Vocabulary, split_identifier_into_parts


BatchedTokenListType = TypeVar("BatchedTokenListType", "BatchedTokenList", "BatchedSubtokenList")
TokenListEmbedderModuleType = TypeVar("TokenListEmbedderModuleType", "TokenEmbedderModule", "SubtokenEmbedderModule")


class TokenListEmbedderModel(
    AbstractBuxlabModel[str, BatchedTokenListType, TokenListEmbedderModuleType],
    Generic[BatchedTokenListType, TokenListEmbedderModuleType],
):
    max_vocabulary_size: int = 10000
    min_freq_threshold: int = 5

    def metadata_init(self) -> Dict[str, Any]:
        return {"counter": Counter()}

    def metadata_process_sample(self, raw_metadata: Dict[str, Any], sample: str) -> None:
        raw_metadata["counter"].update(self.metadata_get_tokens(sample))

    @abstractmethod
    def metadata_get_tokens(self, x: str) -> Iterable[str]:
        raise NotImplementedError

    def metadata_finalize(self, raw_metadata: Dict[str, Any]) -> None:
        self.vocabulary = Vocabulary.create_vocabulary(
            raw_metadata["counter"],
            max_size=self.max_vocabulary_size,
            count_threshold=self.min_freq_threshold,
            add_unk=True,
            add_pad=False,
        )

    def batch_init(self) -> Dict[str, Any]:
        return {"strings": []}

    def batch_can_fit(self, raw_batch: Dict[str, Any], sample: str) -> bool:
        return True

    def batch_add_sample(self, raw_batch: Dict[str, Any], sample: str) -> None:
        raw_batch["strings"].append(sample)

    def batch_add_padding(self, raw_batch: Dict[str, Any], target_num_samples: int) -> None:
        num_samples = len(raw_batch["strings"])
        num_required_pad = target_num_samples - num_samples
        if num_required_pad > 0:
            raw_batch["strings"].extend(self.vocabulary.get_unk() * num_required_pad)

    def batch_pad(self, raw_batch: Dict[str, Any]) -> None:
        num_samples = len(raw_batch["strings"])
        self.batch_add_padding(raw_batch, target_num_samples=self._get_padded_size("strings", num_samples))


class BatchedTokenList(PyTreeNode):
    token_idxs: jnp.ndarray  # int tensor [B]


class TokenEmbedderModule(nn.Module):
    vocabulary_size: int
    embedding_size: int
    dropout_rate: float

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocabulary_size, features=self.embedding_size)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, input: BatchedTokenList, train: bool = False) -> jnp.ndarray:
        return self.dropout(self.embedding(input.token_idxs), deterministic=not train)


@gin.configurable
class TokenEmbedderModel(TokenListEmbedderModel[BatchedTokenList, TokenEmbedderModule]):
    embedding_size: int = 128
    dropout_rate: float = 0.2

    def metadata_get_tokens(self, x: str) -> Iterable[str]:
        return (x,)

    def batch_finalize(self, raw_batch: Dict[str, Any]) -> BatchedTokenList:
        return BatchedTokenList(jnp.array(self.vocabulary.get_id_or_unk_multiple(tokens=raw_batch["strings"])))

    def build_module(self) -> TokenEmbedderModule:
        return TokenEmbedderModule(
            vocabulary_size=len(self.vocabulary),
            embedding_size=self.embedding_size,
            dropout_rate=self.dropout_rate,
        )


class BatchedSubtokenList(PyTreeNode):
    subtoken_idxs: jnp.ndarray  # int tensor [B, max_num_subtokens]
    lengths: jnp.ndarray  # int tensor [B] containing the length of each entry (rest is masked)


class SubtokenEmbedderModule(nn.Module):
    vocabulary_size: int
    embedding_size: int
    subtoken_combination_kind: Literal["mean", "max", "sum"]
    dropout_rate: float
    use_dense_output: bool

    def setup(self) -> None:
        self.embedding = nn.Embed(num_embeddings=self.vocabulary_size, features=self.embedding_size)
        if self.use_dense_output:
            self.dense_output_layer = nn.Dense(features=self.embedding_size, use_bias=False)
        else:
            self.dense_output_layer = None
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, input: BatchedSubtokenList, train: bool = False) -> jnp.ndarray:
        embedded = self.embedding(input.subtoken_idxs)  # [B, max_num_subtokens, D]
        lengths = jnp.expand_dims(input.lengths, 1)
        is_unmasked = jnp.expand_dims(jnp.arange(embedded.shape[1]), 0) < lengths  # [B, max_num_subtokens]

        if self.subtoken_combination_kind == "sum":
            embedded = embedded * jnp.expand_dims(is_unmasked, 1)
            embedded = jnp.sum(embedded, axis=-2)  # [B, D]
        elif self.subtoken_combination_kind == "mean":
            embedded = embedded * jnp.expand_dims(is_unmasked, -1)
            embedded = jnp.sum(embedded, axis=-2) / (lengths + 1e-10)  # [B, D]
        elif self.subtoken_combination_kind == "max":
            embedded = jnp.where(is_unmasked, embedded, -jnp.inf)
            embedded = jnp.max(embedded, axis=-2)  # [B, D]
        else:
            raise ValueError(f'Unrecognized subtoken combination "{self.subtoken_combination_kind}".')

        if self.dense_output_layer is not None:
            embedded = self.dense_output_layer(embedded)

        return self.dropout(embedded, deterministic=not train)


@gin.configurable
@dataclass
class SubtokenEmbedderModel(TokenListEmbedderModel[BatchedSubtokenList, SubtokenEmbedderModule]):
    embedding_size: int = 128
    subtoken_combination_kind: Literal["mean", "max", "sum"] = "mean"
    max_num_subtokens: int = 5
    dropout_rate: float = 0.2
    use_dense_output: bool = True

    def metadata_get_tokens(self, x: str) -> List[str]:
        return split_identifier_into_parts(x)[: self.max_num_subtokens]

    def batch_finalize(self, raw_batch: Dict[str, Any]) -> BatchedSubtokenList:
        subtoken_ids, lengths = [], []
        unk_id = self.vocabulary.get_id_or_unk(self.vocabulary.get_unk())
        for s in raw_batch["strings"]:
            subtokens = self.metadata_get_tokens(s)
            length = len(subtokens)
            padding = self.max_num_subtokens - length
            subtoken_ids.append(self.vocabulary.get_id_or_unk_multiple(subtokens) + [unk_id] * padding)
            lengths.append(length)

        return BatchedSubtokenList(subtoken_idxs=jnp.array(subtoken_ids), lengths=jnp.array(lengths))

    def build_module(self) -> SubtokenEmbedderModule:
        return SubtokenEmbedderModule(
            vocabulary_size=len(self.vocabulary),
            embedding_size=self.embedding_size,
            subtoken_combination_kind=self.subtoken_combination_kind,
            dropout_rate=self.dropout_rate,
            use_dense_output=self.use_dense_output,
        )
