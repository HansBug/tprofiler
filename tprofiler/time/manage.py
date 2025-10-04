import time
from contextlib import contextmanager
from dataclasses import field, dataclass
from typing import Dict, List

import torch
import torch.distributed as dist

from ..distribution import gather


@dataclass
class TimeManager:
    records: Dict[str, List[float]] = field(default_factory=dict)

    def _append_time(self, name: str, secs: float):
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(secs)

    def _get_time(self, name: str) -> List[float]:
        if name not in self.records:
            return []
        else:
            return self.records[name]

    def _get_time_torch(self, name: str) -> torch.Tensor:
        return torch.tensor(self._get_time(name), dtype=torch.float32)

    @contextmanager
    def timer(self, name: str):
        start_time = time.time()
        yield
        self._append_time(name, time.time() - start_time)

    def clear(self):
        self.records.clear()

    def gather(self):
        retval = {}
        for key in self.records:
            times = self._get_time_torch(key)
            gathered_times = gather(times, dst=0)
            if gathered_times is not None:
                retval[key] = gathered_times

        if dist.get_rank() == 0:
            return retval
        else:
            return None
