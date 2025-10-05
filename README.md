# tprofiler

Torch Profiler (Based on Verl Project)

## Time profiler and gather

```python
import os
import random
import time
from pprint import pprint

import torch.distributed as dist
import torch.multiprocessing as mp

from tprofiler.time import timer_wrap, TimeManager


# Example function decorated with timer_wrap for automatic timing
# The @timer_wrap() decorator enables this function to be automatically timed
# when called within a TimeManager's enable_timer() context
@timer_wrap()
def task_a():
    """Simulates a task that takes 0.1-0.5 seconds to complete"""
    time.sleep(random.uniform(0.1, 0.5))


# Another example function with timer_wrap decorator
@timer_wrap()
def task_b():
    """Simulates a task that takes 0.2-0.8 seconds to complete"""
    time.sleep(random.uniform(0.2, 0.8))


def worker(rank, world_size):
    """
    Worker process function that demonstrates tprofiler timing capabilities

    Args:
        rank: Process rank in distributed setup
        world_size: Total number of processes
    """
    # Set up distributed environment for multi-process communication
    os.environ.update({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12355'})
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    # Create a TimeManager instance to handle timing operations
    # TimeManager is the core component that collects and manages timing data
    tm = TimeManager()

    # Method 1: Manual timing using context manager
    # Use tm.timer() to manually time a specific code block
    # The 'name' parameter allows you to give a custom name to this timing section
    with tm.timer(name='xxxx'):
        # Any code within this context will be timed and recorded
        time.sleep(random.uniform(0.05, 0.2))

    # Method 2: Automatic timing of decorated functions
    # When enable_timer() is active, all functions decorated with @timer_wrap()
    # will be automatically timed and their execution times recorded
    with tm.enable_timer():
        task_a()  # This call will be timed and recorded
        task_b()  # This call will be timed and recorded
        task_a()  # This call will also be timed and recorded

    # Function calls outside enable_timer() context are NOT recorded
    # Even though task_b has @timer_wrap(), this call won't be timed
    task_b()

    # Collect timing data from all processes and display results
    # tm.gather() collects timing statistics from the current process
    # The output shows: (process_rank, timing_data_dictionary)
    pprint((dist.get_rank(), tm.gather()))

    # Clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    print("Starting 4-process timing example...")
    print("This demo shows how tprofiler works in a multi-process environment")
    print("Each process will:")
    print("1. Time a custom code block using tm.timer()")
    print("2. Time decorated functions using tm.enable_timer()")
    print("3. Gather and display timing statistics")
    print("-" * 50)

    # Spawn 4 worker processes to demonstrate multi-process timing
    # mp.spawn creates multiple processes, each running the worker function
    # args=(4,) passes world_size=4 to each worker
    # nprocs=4 specifies the number of processes to create
    # join=True waits for all processes to complete before continuing
    mp.spawn(worker, args=(4,), nprocs=4, join=True)

    print("-" * 50)
    print("Demo completed! Check the output above to see timing data from each process.")
    print("Key features demonstrated:")
    print("- Manual timing with tm.timer(name='...')")
    print("- Automatic timing with @timer_wrap() + tm.enable_timer()")
    print("- Multi-process timing data collection")
```

```
Starting 4-process timing example...
This demo shows how tprofiler works in a multi-process environment
Each process will:
1. Time a custom code block using tm.timer()
2. Time decorated functions using tm.enable_timer()
3. Gather and display timing statistics
--------------------------------------------------
[Gloo] Rank 0 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 1 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 3 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 2 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
(1, None)
(2, None)
(3, None)
(0,
 {'task_a': tensor([0.4433, 0.4100, 0.3646, 0.4867, 0.2926, 0.1256, 0.4653, 0.2073]),
  'task_b': tensor([0.3920, 0.4725, 0.6063, 0.3331]),
  'xxxx': tensor([0.1618, 0.1673, 0.0963, 0.0562])})
--------------------------------------------------
Demo completed! Check the output above to see timing data from each process.
Key features demonstrated:
- Manual timing with tm.timer(name='...')
- Automatic timing with @timer_wrap() + tm.enable_timer()
- Multi-process timing data collection
```
