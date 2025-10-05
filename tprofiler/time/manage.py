import time
from contextlib import contextmanager
from dataclasses import field, dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed as dist
from hbutils.reflection import context

from ..distribution import gather
from ..utils import Stack

_TIMER_STACK_NAME = 'timer_stack'


def _get_timer_stack() -> Stack:
    timer_stack = context().get(_TIMER_STACK_NAME, None) or Stack()
    return timer_stack


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

    def _get_time_with_rank(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self._get_time_torch(name)
        r = torch.ones_like(t, dtype=torch.int32) * dist.get_rank()
        return t, r

    @contextmanager
    def timer(self, name: str):
        start_time = time.time()
        yield
        self._append_time(name, time.time() - start_time)

    def clear(self):
        self.records.clear()

    @contextmanager
    def enable_timer(self):
        timer_stack = _get_timer_stack()
        timer_stack.push(self)
        try:
            with context().vars(**{_TIMER_STACK_NAME: timer_stack}):
                yield
        finally:
            timer_stack.pop()

    def gather(self, dst: Optional[int] = None) -> Optional[Dict[str, 'GatheredTime']]:
        is_gather_dst = dst is None or dst == dist.get_rank()
        retval = {} if is_gather_dst else None
        for key in self.records:
            times, ranks = self._get_time_with_rank(key)
            gathered_times = gather(times, dst=dst)
            gathered_ranks = gather(ranks, dst=dst)
            if is_gather_dst:
                retval[key] = GatheredTime(
                    times=gathered_times,
                    ranks=gathered_ranks,
                )

        return retval


@dataclass
class GatheredTime:
    times: torch.Tensor
    ranks: torch.Tensor

    def get_rank(self, rank: int) -> 'GatheredTime':
        f = self.ranks == rank
        return GatheredTime(
            times=self.times[f],
            ranks=self.ranks[f],
        )

    def sum(self) -> float:
        return self.times.sum().detach().cpu().item()

    def count(self) -> int:
        return self.times.shape[0]

    def mean(self) -> float:
        return self.times.mean().detach().cpu().item()

    def std(self) -> float:
        return self.times.std().detach().cpu().item()
