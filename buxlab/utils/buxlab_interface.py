from dataclasses import dataclass
from typing import Dict, Any, Generic, TypeVar, Iterable, Iterator
from abc import ABC, abstractmethod

import flax.linen as nn
from absl import logging

from .misc import round_to_nearest_mult_of_power_of_2

RawDataType = TypeVar("RawDataType")
BatchedDataType = TypeVar("BatchedDataType")
ReturnType = TypeVar("ReturnType")
ModuleType = TypeVar("ModuleType", bound=nn.Module)


@dataclass
class AbstractBuxlabModel(ABC, Generic[RawDataType, BatchedDataType, ModuleType]):
    pad_batches: bool = True

    @abstractmethod
    def metadata_init(self) -> Dict[str, Any]:
        """
        Returns:
            Dict used by the implementor to collect metadata (e.g., words to build a vocabulary)
        """
        raise NotImplementedError

    @abstractmethod
    def metadata_process_sample(self, raw_metadata: Dict[str, Any], sample: RawDataType) -> None:
        """
        Args:
            raw_metadata: dict as returned by `metadata_init`, should be updated appropriately.
            sample: sample to process.
        """
        raise NotImplementedError

    @abstractmethod
    def metadata_finalize(self, raw_metadata: Dict[str, Any]) -> None:
        """
        Finalize and store processed metadata (e.g., a vocabulary).

        Args:
            raw_metadata: dict as returned by `metadata_init` and updated by `metadata_process_sample`
        """
        raise NotImplementedError

    def compute_metadata(self, samples: Iterable[RawDataType]) -> None:
        cur_metadata = self.metadata_init()
        for sample in samples:
            self.metadata_process_sample(cur_metadata, sample)
        self.metadata_finalize(cur_metadata)

    @abstractmethod
    def batch_init(self) -> Dict[str, Any]:
        """
        Returns:
            Dict used by the implementor to collect data for a batch (e.g., nodes of different graphs)
        """
        raise NotImplementedError

    @abstractmethod
    def batch_can_fit(self, raw_batch: Dict[str, Any], sample: RawDataType) -> bool:
        """
        Return True if adding this sample does not violate any size bounds on the batch.

        Args:
            raw_batch: dict as returned by `batching_init`, should be updated appropriately.
            sample: sample to fit
        """
        raise NotImplementedError

    @abstractmethod
    def batch_add_sample(self, raw_batch: Dict[str, Any], sample: RawDataType) -> None:
        """
        Args:
            raw_batch: dict as returned by `batching_init`, should be updated appropriately.
            sample: sample to process.
        """
        raise NotImplementedError

    def _get_padded_size(self, name: str, cur_size: int) -> int:
        """
        Args:
            name: name of the element for which we are trying to get a padding bound
            cur_size: current (i.e., minimal) size of the element we are interested in padding

        Returns:
            A value that is at least `cur_size`.
        """
        assert self.pad_batches

        padding_bounds: Dict[str, int] = getattr(self, "_padding_bounds", {})
        setattr(self, "_padding_bounds", padding_bounds)

        old_size = padding_bounds.get(name, 0)
        new_size = round_to_nearest_mult_of_power_of_2(max(cur_size, old_size))
        padding_bounds[name] = new_size
        if old_size != new_size:  # TODO: remove
            logging.debug(f"Updating padding bound for {name} from {old_size} to {new_size}")
        return new_size

    @abstractmethod
    def batch_pad(self, raw_batch: Dict[str, Any]) -> None:
        """
        Extend `raw_batch`

        Args:
            raw_batch: dict as returned by `batching_init`, should be updated appropriately.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_finalize(self, raw_batch: Dict[str, Any]) -> BatchedDataType:
        """
        Args:
            raw_batch: dict as returned by `batching_init`, should be updated appropriately.

        Returns:
            batched_data, such that `__call__(batched_data, ...)` succeeds
        """
        raise NotImplementedError

    @abstractmethod
    def build_module(self) -> ModuleType:
        raise NotImplementedError

    def batch_iterator(self, samples: Iterable[RawDataType]) -> Iterator[BatchedDataType]:
        cur_batch, cur_batch_empty = self.batch_init(), True
        for sample in samples:
            if not self.batch_can_fit(cur_batch, sample):
                self.batch_pad(cur_batch)
                yield self.batch_finalize(cur_batch)
                cur_batch, cur_batch_empty = self.batch_init(), True

            self.batch_add_sample(cur_batch, sample)
            cur_batch_empty = False

        if not cur_batch_empty:
            self.batch_pad(cur_batch)
            yield self.batch_finalize(cur_batch)
