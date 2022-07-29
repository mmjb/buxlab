from dataclasses import dataclass
from functools import partial
import logging
from typing import Any, Iterable, Dict, List
from pathlib import Path

import numpy as np

from .msgpackutils import load_msgpack_l_gz
from .file_reader_iterable import SequentialFileReaderIterable


logger = logging.getLogger(__name__)


def read_msgpack_l_gz_chunk(paths: List[Path], chunk_idx: int, seed: int = 0, shuffle: bool = False):
    data: List[Dict[str, Any]] = []
    for path in paths:
        data.extend(load_msgpack_l_gz(path))
    if shuffle:
        rng = np.random.default_rng(seed + chunk_idx)
        rng.shuffle(data)
    yield from data


@dataclass(frozen=True)
class BuglabDataset:
    data_folder: Path

    def _get_fold_files(self, fold: str) -> List[Path]:
        return list(self.data_folder.glob(f"{fold}/*.msgpack.l.gz"))

    def __post_init__(self):
        logger.info(f"Created dataset backed by {self.data_folder}.")
        for fold in ("train", "valid", "test", "test-only"):
            num_files = len(self._get_fold_files(fold))
            logger.info(f"  fold {fold:10s}: {num_files}")

    def get_sample_iterable(
        self,
        fold: str,
        seed: int = 0,
        repeat: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        # TODO: should be BufferedFileReaderIterable
        shuffle_data = fold == "train"
        return SequentialFileReaderIterable(
            reader_fn=partial(read_msgpack_l_gz_chunk, seed=seed, shuffle=shuffle_data),
            data_paths=self._get_fold_files(fold),
            shuffle_data=shuffle_data,
            repeat=repeat,
        )
