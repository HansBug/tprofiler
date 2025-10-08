# tprofiler

Torch Profiler (Based on Verl Project)

## Time profiler and gather

```python
import os
import random
import time

import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from tprofiler.time import timer_wrap, TimeManager, timer, ProfiledTime


# Example function decorated with timer_wrap for automatic timing
# The @timer_wrap() decorator enables this function to be automatically timed
# when called within a TimeManager's enable_timer() context
@timer_wrap()
def task_a():
    """Simulates a task that takes 0.1-0.5 seconds to complete"""
    time.sleep(random.uniform(0.1, 0.5))


# Another example function with timer_wrap decorator
# You can optionally specify a custom name for the timing measurement
@timer_wrap('custom_task_b')
def task_b():
    """Simulates a task that takes 0.2-0.8 seconds to complete with nested timing"""
    # Use the timer context manager for nested timing within the function
    with timer('task_b_inner'):
        time.sleep(random.uniform(0.2, 0.8))


def worker(rank, world_size):
    """
    Worker process function that demonstrates tprofiler timing capabilities
    in a distributed environment.

    Args:
        rank (int): Process rank in distributed setup (0 to world_size-1)
        world_size (int): Total number of processes in the distributed group
    """
    # Set up distributed environment for multi-process communication
    # This enables timing data to be gathered across all processes
    os.environ.update({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12355'})
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    print(f"Process {rank}/{world_size} started")

    # Create a TimeManager instance to handle timing operations
    # TimeManager is the core component that collects and manages timing data
    tm = TimeManager()

    # Method 1: Manual timing using context manager
    # Use tm.timer() to manually time a specific code block
    # The 'name' parameter allows you to give a custom name to this timing section
    with tm.timer(name='manual_timing_block'):
        # Any code within this context will be timed and recorded under 'manual_timing_block'
        time.sleep(random.uniform(0.05, 0.2))

    # Method 2: Automatic timing of decorated functions
    # When enable_timer() is active, all functions decorated with @timer_wrap()
    # will be automatically timed and their execution times recorded
    with tm.enable_timer():
        # All these function calls will be automatically timed and recorded
        task_a()  # Recorded under 'task_a' (function name)
        task_b()  # Recorded under 'custom_task_b' (custom name from decorator)

        # Multiple calls to the same function accumulate timing data
        for i in range(5):  # Reduced from 10 for faster demo
            task_a()  # Each call adds another timing measurement to 'task_a'

        # Method 3: Using timer context manager within enable_timer context
        # This works with both the active TimeManager and any in the timer stack
        with timer('loop_operation'):
            for i in range(3):
                time.sleep(random.uniform(0.01, 0.05))

    # Function calls outside enable_timer() context are NOT automatically recorded
    # Even though task_b has @timer_wrap(), this call won't be timed by the TimeManager
    task_b()  # This call is NOT recorded in tm.records

    print(f"Process {rank} completed timing operations")

    # Gather timing data from all processes to rank 0 for analysis
    # tm.gather(dst=0) collects timing data from all processes to rank 0
    # Returns None for non-destination ranks, GatheredTime objects for destination rank
    gathered_data = tm.gather(dst=0)

    if gathered_data is not None:
        # Save the gathered timing data to a file for persistence
        # The .pt extension indicates this is a PyTorch tensor file format
        # This allows the profiling results to be stored and analyzed later
        gathered_data.save('test_exported_time.pt')

        # Demonstrate loading previously saved timing data
        # ProfiledTime.load() can restore timing data from disk
        # Note: The file extension is automatically handled (.pt is added if not present)
        loaded_gathered_data = ProfiledTime.load('test_exported_time')

        # Display the loaded data structure for verification
        # This shows that the save/load cycle preserves all timing information
        # The !r format specifier uses the object's __repr__ method for detailed output
        print(f'Loaded gathered data: {loaded_gathered_data!r}')

        # Only rank 0 will have gathered_data (not None)
        print(f"\n=== Timing Analysis (Rank {rank}) ===")

        # Analyze task_a timing data
        if 'task_a' in gathered_data:
            task_a_data = gathered_data['task_a']
            print(f"Task A Statistics:")
            print(f"  - Total time: {task_a_data.sum():.3f}s")
            print(f"  - Count: {task_a_data.count()} measurements")
            print(f"  - Mean: {task_a_data.mean():.3f}s")
            print(f"  - Std: {task_a_data.std():.3f}s")
            print(f"  - Ranks involved: {task_a_data.rank_count()}")

            # Create histogram visualization
            try:
                plt.figure(figsize=(10, 6))
                task_a_data.hist(separate_ranks=True, title='Task A Execution Times', bins=10)
                plt.tight_layout()
                plt.savefig('task_a_timing.png', dpi=150, bbox_inches='tight')
                plt.show()
                print("  - Histogram saved as 'task_a_timing.png'")
            except Exception as e:
                print(f"  - Could not create histogram: {e}")

        # Analyze other timing data
        for name, data in gathered_data.items():
            if name != 'task_a':  # Already analyzed above
                print(f"\n{name.replace('_', ' ').title()} Statistics:")
                print(f"  - Total time: {data.sum():.3f}s")
                print(f"  - Count: {data.count()} measurements")
                print(f"  - Mean: {data.mean():.3f}s")
                print(f"  - Std: {data.std():.3f}s")
    else:
        print(f"Process {rank} - timing data sent to rank 0 for analysis")

    # Clean up distributed process group
    dist.destroy_process_group()
    print(f"Process {rank} finished")


if __name__ == "__main__":
    print("=" * 60)
    print("TPROFILER DISTRIBUTED TIMING DEMO")
    print("=" * 60)
    print("This demo demonstrates tprofiler's capabilities in a multi-process environment.")
    print("\nFeatures showcased:")
    print("1. Manual timing with tm.timer(name='...')")
    print("2. Automatic timing with @timer_wrap() + tm.enable_timer()")
    print("3. Nested timing with timer() context manager")
    print("4. Multi-process timing data collection and analysis")
    print("5. Statistical analysis and visualization of timing data")
    print("\nStarting 4 worker processes...")
    print("-" * 60)

    # Spawn 4 worker processes to demonstrate multi-process timing
    # mp.spawn creates multiple processes, each running the worker function
    # args=(4,) passes world_size=4 to each worker process
    # nprocs=4 specifies the number of processes to create
    # join=True waits for all processes to complete before continuing
    mp.spawn(worker, args=(4,), nprocs=4, join=True)

    print("-" * 60)
    print("✅ Demo completed successfully!")
    print("\nKey takeaways:")
    print("• TimeManager.timer() - Manual timing of code blocks")
    print("• @timer_wrap() - Automatic timing of decorated functions")
    print("• TimeManager.enable_timer() - Activates automatic timing")
    print("• timer() - Context manager for timing arbitrary code")
    print("• TimeManager.gather() - Collects timing data across processes")
    print("• GatheredTime - Provides analysis and visualization tools")
```
