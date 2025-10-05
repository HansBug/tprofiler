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


@timer_wrap()
def task_a():
    time.sleep(random.uniform(0.1, 0.5))


@timer_wrap()
def task_b():
    time.sleep(random.uniform(0.2, 0.8))


def worker(rank, world_size):
    """工作进程"""
    os.environ.update({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12355'})
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    tm = TimeManager()

    # 模拟一些任务
    with tm.enable_timer():
        task_a()
        task_b()
        task_a()

    task_b()

    pprint((dist.get_rank(), tm.gather()))
    dist.destroy_process_group()


if __name__ == "__main__":
    print("Starting 4-process timing example...")
    mp.spawn(worker, args=(4,), nprocs=4, join=True)
    print("Done!")
```

```
Starting 4-process timing example...
[Gloo] Rank 0 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 2 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 1 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 3 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
(1, None)
(2, None)
(3, None)
(0,
 {'task_a': tensor([0.1167, 0.2070, 0.3979, 0.1128, 0.3321, 0.3045, 0.3024, 0.2376]),
  'task_b': tensor([0.2139, 0.3364, 0.4123, 0.6122])})
Done!
```
