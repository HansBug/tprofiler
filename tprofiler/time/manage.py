"""
Time measurement and profiling utilities for distributed PyTorch applications.

This module provides comprehensive timing functionality for distributed training scenarios,
including context managers for timing code blocks, gathering timing data across processes,
and visualization tools for analyzing performance metrics.

The main components include:

- TimeManager: Core timing functionality with context managers
- GatheredTime: Analysis and visualization of distributed timing data
"""

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
    """
    Get the current timer stack from the context.

    This function retrieves the timer stack from the current execution context,
    creating a new stack if none exists. The timer stack manages active
    TimeManager instances for nested timing operations.

    :return: The timer stack instance.
    :rtype: Stack
    """
    timer_stack = context().get(_TIMER_STACK_NAME, None) or Stack()
    return timer_stack


@dataclass
class TimeManager:
    """
    A comprehensive time measurement manager for tracking execution times.

    This class provides functionality to measure and record execution times for different
    operations, with support for distributed environments and context management.
    It maintains timing records for named operations and provides context managers
    for convenient timing measurement.

    :param records: Dictionary storing timing records for different operations.
    :type records: Dict[str, List[float]]

    Example::

        >>> tm = TimeManager()
        >>> with tm.timer('operation1'):
        ...     time.sleep(0.1)
        >>> times = tm._get_time('operation1')
        >>> len(times) == 1
        True
    """
    records: Dict[str, List[float]] = field(default_factory=dict)

    def _append_time(self, name: str, secs: float):
        """
        Append a timing record for the specified operation.

        This method adds a new timing measurement to the records for the given
        operation name. If this is the first measurement for the operation,
        a new list is created.

        :param name: The name of the operation being timed.
        :type name: str
        :param secs: The execution time in seconds.
        :type secs: float
        """
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(secs)

    def _get_time(self, name: str) -> List[float]:
        """
        Get all timing records for a specific operation.

        Retrieves the complete list of timing measurements recorded for the
        specified operation name. Returns an empty list if no measurements
        have been recorded for this operation.

        :param name: The name of the operation.
        :type name: str
        :return: List of timing records in seconds.
        :rtype: List[float]
        """
        if name not in self.records:
            return []
        else:
            return self.records[name]

    def _get_time_torch(self, name: str) -> torch.Tensor:
        """
        Get timing records as a PyTorch tensor.

        Converts the timing records for the specified operation into a
        PyTorch tensor with float32 dtype. This is useful for mathematical
        operations and distributed gathering.

        :param name: The name of the operation.
        :type name: str
        :return: Timing records as a float32 tensor.
        :rtype: torch.Tensor
        """
        return torch.tensor(self._get_time(name), dtype=torch.float32)

    def _get_time_with_rank(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get timing records along with corresponding rank information.

        Returns timing data paired with rank information for distributed
        environments. Each timing measurement is associated with the current
        process rank.

        :param name: The name of the operation.
        :type name: str
        :return: Tuple of (times, ranks) tensors.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        t = self._get_time_torch(name)
        r = torch.ones_like(t, dtype=torch.int32) * dist.get_rank()
        return t, r

    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing code execution.

        This context manager measures the execution time of the code block
        within it and automatically records the timing under the specified name.
        It uses high-precision time measurement and is suitable for both
        short and long-running operations.

        :param name: The name to associate with this timing measurement.
        :type name: str
        :yields: None

        Example::

            >>> tm = TimeManager()
            >>> with tm.timer('my_operation'):
            ...     # Your code here
            ...     pass
        """
        start_time = time.time()
        yield
        self._append_time(name, time.time() - start_time)

    def clear(self):
        """
        Clear all timing records.

        Removes all recorded timing data from this TimeManager instance,
        effectively resetting it to a clean state.
        """
        self.records.clear()

    @contextmanager
    def enable_timer(self):
        """
        Context manager to enable this TimeManager in the timer stack.

        This method pushes the current TimeManager instance onto the timer stack,
        making it available for use by timer decorators and context managers
        throughout the nested execution context. The TimeManager is automatically
        removed from the stack when the context exits.

        :yields: None

        Example::

            >>> tm = TimeManager()
            >>> with tm.enable_timer():
            ...     # Now timer decorators and contexts will use this TimeManager
            ...     with timer('operation'):
            ...         pass
        """
        timer_stack = _get_timer_stack()
        timer_stack.push(self)
        try:
            with context().vars(**{_TIMER_STACK_NAME: timer_stack}):
                yield
        finally:
            timer_stack.pop()

    def gather(self, dst: Optional[int] = None) -> Optional[Dict[str, 'GatheredTime']]:
        """
        Gather timing data from all processes in a distributed environment.

        Collects timing measurements from all processes in the distributed group
        and returns them as GatheredTime objects for analysis. This is useful
        for understanding performance characteristics across the entire distributed
        training setup.

        :param dst: Destination rank to gather data to. If None, gathers to all ranks.
        :type dst: Optional[int]
        :return: Dictionary of gathered timing data, or None if not the destination rank.
        :rtype: Optional[Dict[str, GatheredTime]]

        Example::

            >>> tm = TimeManager()
            >>> # After recording some times...
            >>> gathered = tm.gather(dst=0)  # Gather to rank 0
        """
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
    """
    Container for timing data gathered from multiple processes.

    This class provides analysis and visualization capabilities for timing data
    collected from distributed training environments. It maintains timing
    measurements along with their corresponding process ranks, enabling
    detailed performance analysis across the distributed system.

    :param times: Tensor containing timing measurements.
    :type times: torch.Tensor
    :param ranks: Tensor containing corresponding rank information.
    :type ranks: torch.Tensor

    Example::

        >>> times = torch.tensor([0.1, 0.2, 0.15])
        >>> ranks = torch.tensor([0, 1, 0])
        >>> gt = GatheredTime(times, ranks)
        >>> gt.mean()
        0.15...
    """
    times: torch.Tensor
    ranks: torch.Tensor

    def get_rank(self, *ranks: int) -> 'GatheredTime':
        """
        Filter timing data for specific ranks.

        Creates a new GatheredTime instance containing only the timing data
        from the specified process ranks. This is useful for analyzing
        performance characteristics of specific processes or comparing
        performance across different ranks.

        :param ranks: Rank numbers to filter for.
        :type ranks: int
        :return: New GatheredTime instance with filtered data.
        :rtype: GatheredTime

        Example::

            >>> gt = GatheredTime(torch.tensor([0.1, 0.2]), torch.tensor([0, 1]))
            >>> rank0_data = gt.get_rank(0)
        """
        mask = torch.zeros_like(self.ranks, dtype=torch.bool, device=self.ranks.device)
        for rank in ranks:
            mask |= (self.ranks == rank)
        return GatheredTime(
            times=self.times[mask],
            ranks=self.ranks[mask],
        )

    def __getitem__(self, item) -> 'GatheredTime':
        """
        Get a subset of the gathered time data using indexing.

        Supports standard Python indexing and slicing operations to extract
        subsets of the timing data while maintaining the correspondence
        between times and ranks.

        :param item: Index or slice to apply to the data.
        :return: New GatheredTime instance with indexed data.
        :rtype: GatheredTime
        """
        return GatheredTime(
            times=self.times[item],
            ranks=self.ranks[item],
        )

    def __bool__(self):
        """
        Check if the gathered time data is non-empty.

        Returns True if there are timing measurements available, False if
        the data structure is empty.

        :return: True if there is timing data, False otherwise.
        :rtype: bool
        """
        return self.ranks.numel() > 0

    def sum(self) -> float:
        """
        Calculate the sum of all timing measurements.

        Computes the total time across all measurements, which can be useful
        for understanding the cumulative time spent on an operation across
        all processes.

        :return: Sum of all times in seconds.
        :rtype: float
        """
        return self.times.sum().detach().cpu().item()

    def count(self) -> int:
        """
        Get the total number of timing measurements.

        Returns the total count of timing measurements across all ranks,
        which indicates how many times the measured operation was executed.

        :return: Number of timing measurements.
        :rtype: int
        """
        return self.times.shape[0]

    def mean(self) -> float:
        """
        Calculate the mean of all timing measurements.

        Computes the average execution time across all measurements and ranks,
        providing a central tendency measure for the operation performance.

        :return: Mean time in seconds.
        :rtype: float
        """
        return self.times.mean().detach().cpu().item()

    def std(self) -> float:
        """
        Calculate the standard deviation of timing measurements.

        Computes the standard deviation to measure the variability in
        execution times, which can indicate performance consistency
        across processes and executions.

        :return: Standard deviation in seconds.
        :rtype: float
        """
        return self.times.std().detach().cpu().item()

    def rank_count(self) -> int:
        """
        Get the number of unique ranks in the data.

        Returns the count of distinct process ranks represented in the
        gathered timing data, indicating how many processes contributed
        measurements.

        :return: Number of unique ranks.
        :rtype: int
        """
        return torch.unique(self.ranks).numel()

    def hist(self, bins: Optional[int] = None, separate_ranks: bool = False,
             alpha: float = 0.7, title: Optional[str] = None, ax=None, **kwargs):
        """
        Plot histogram of timing distribution.

        Creates a histogram visualization of the timing data distribution,
        with options to show all ranks together or separately. This is useful
        for understanding the performance characteristics and identifying
        outliers or patterns in execution times.

        :param bins: Number of histogram bins. If None, uses matplotlib default.
        :type bins: Optional[int]
        :param separate_ranks: Whether to show each rank separately.
        :type separate_ranks: bool
        :param alpha: Transparency level when separate_ranks=True.
        :type alpha: float
        :param title: Title for the plot. Defaults to 'Time Distribution'.
        :type title: Optional[str]
        :param ax: Matplotlib axes object for plotting. If None, uses current axes.
        :type ax: Optional[matplotlib.axes.Axes]
        :param kwargs: Additional arguments passed to matplotlib hist function.
        :type kwargs: dict

        Example::

            >>> import matplotlib.pyplot as plt
            >>> gt = GatheredTime(torch.tensor([0.1, 0.2, 0.15]), torch.tensor([0, 1, 0]))
            >>> fig, ax = plt.subplots()
            >>> gt.hist(ax=ax, bins=10)
        """
        ax = ax or plt.gca()
        title = title or 'Time Distribution'
        times_np = self.times.detach().cpu().numpy()
        ranks_np = self.ranks.detach().cpu().numpy()

        unique_ranks = np.unique(ranks_np)
        if not separate_ranks:
            # Show all ranks together
            ax.hist(times_np, bins=bins, alpha=1.0, **kwargs)
            ax.set_title(f'{title}\n'
                         f'(All {plural_word(self.rank_count(), "Rank")}, n={len(times_np)}, '
                         f'mean={self.mean():.3g}s, std={self.std():.3g}s)')
        else:
            # Show each rank separately
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
