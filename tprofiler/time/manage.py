import time
from contextlib import contextmanager
from dataclasses import field, dataclass
from typing import Dict, List

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
