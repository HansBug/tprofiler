import time
from contextlib import contextmanager
from dataclasses import field, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
from hbutils.reflection import context
from hbutils.string import plural_word
from matplotlib import pyplot as plt

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

    def get_rank(self, *ranks: int) -> 'GatheredTime':
        mask = torch.zeros_like(self.ranks, dtype=torch.bool, device=self.ranks.device)
        for rank in ranks:
            mask |= (self.ranks == rank)
        return GatheredTime(
            times=self.times[mask],
            ranks=self.ranks[mask],
        )

    def __bool__(self):
        return self.ranks.numel() > 0

    def sum(self) -> float:
        return self.times.sum().detach().cpu().item()

    def count(self) -> int:
        return self.times.shape[0]

    def mean(self) -> float:
        return self.times.mean().detach().cpu().item()

    def std(self) -> float:
        return self.times.std().detach().cpu().item()

    def rank_count(self) -> int:
        return torch.unique(self.ranks).numel()

    def hist(self, bins: Optional[int] = None, separate_ranks: bool = False,
             alpha: float = 0.7, title: Optional[str] = None, ax=None, **kwargs):
        """
        绘制时间分布的直方图

        Args:
            ax: matplotlib的Axes对象，用于绘制
            bins: 直方图的bins数量，默认30
            separate_ranks: 是否将每个rank分开展示，默认False
            alpha: 透明度，当separate_ranks=True时有用，默认0.7
            **kwargs: 传递给ax.hist的其他参数
        """
        ax = ax or plt.gca()
        title = title or 'Time Distribution'
        times_np = self.times.detach().cpu().numpy()
        ranks_np = self.ranks.detach().cpu().numpy()

        unique_ranks = np.unique(ranks_np)
        if not separate_ranks:
            # 所有rank放在一起展示
            ax.hist(times_np, bins=bins, alpha=1.0, **kwargs)
            ax.set_title(f'{title}\n'
                         f'(All {plural_word(self.rank_count(), "Rank")}, n={len(times_np)}, '
                         f'mean={self.mean():.3g}s, std={self.std():.3g}s)')
        else:
            # 每个rank分开展示
            for i, rank in enumerate(unique_ranks):
                rank_mask = ranks_np == rank
                rank_times = times_np[rank_mask]
                ax.hist(
                    rank_times, bins=bins, alpha=alpha,
                    label=f'Rank #{rank} (n={len(rank_times)}, '
                          f'mean={rank_times.mean():.2g}s, std={rank_times.std():.2g}s)',
                    **kwargs
                )

            ax.set_title(f'{title} by {plural_word(self.rank_count(), "Rank")}\n'
                         f'(mean={self.mean():.3g}s, std={self.std():.3g}s)')
            ax.legend()

        ax.set_xlabel(f'Time (seconds)')
        ax.set_ylabel(f'Frequency')
        ax.grid()
