from collections import defaultdict
from typing import Optional, Dict, List

import numpy as np
from absl import logging
from flax.metrics import tensorboard


class MetricLogger:
    def __init__(
        self,
        log_every_num_steps: int = 10,
        log_prefix: str = "",
        tb_writer: Optional[tensorboard.SummaryWriter] = None,
    ):
        self._log_every_num_steps = log_every_num_steps
        self._logged_metrics: Dict[str, List[float]] = defaultdict(list)
        self._step_counter = 0
        self._log_prefix = log_prefix
        self._tb_writer = tb_writer

    def log(self, **metrics):
        for k, v in metrics.items():
            self._logged_metrics[k].append(v)

        self._step_counter += 1

        if self._step_counter % self._log_every_num_steps == 0:
            window_means = {}
            for k, v in self._logged_metrics.items():
                mean_v = np.mean(v)
                window_means[k] = mean_v
                if self._tb_writer is not None:
                    self._tb_writer.scalar(
                        tag=f"{self._log_prefix}{k}",
                        value=mean_v,
                        step=self._step_counter,
                    )
                v.clear()
            logging.info(
                "%sstep % 5d. Avgs since last: %s",
                self._log_prefix,
                self._step_counter,
                ", ".join(f"{k}: {v:.4f}" for k, v in window_means.items()),
            )
