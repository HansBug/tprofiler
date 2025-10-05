"""
Timer decorators and context managers for distributed PyTorch applications.

This module provides convenient decorators and context managers for timing function
execution and code blocks. It integrates with the TimeManager system to enable
seamless timing across distributed environments.

The main components include:

- timer_wrap: Decorator for automatically timing function execution
- timer: Context manager for timing arbitrary code blocks
"""

from contextlib import contextmanager
from functools import wraps
from typing import Optional

from hbutils.reflection import nested_with

from .manage import _get_timer_stack


def timer_wrap(name: Optional[str] = None):
    """
    Decorator that automatically times function execution using the active timer stack.

    This decorator wraps a function to measure its execution time and record it
    using all active TimeManager instances in the timer stack. The timing name
    defaults to the function's name if not specified.

    :param name: Custom name for the timing measurement. If None, uses function name.
    :type name: Optional[str]
    :return: Decorator function that wraps the target function with timing.
    :rtype: callable

    Example::

        >>> @timer_wrap('my_function')
        ... def slow_function():
        ...     import time
        ...     time.sleep(0.1)
        ...     return "done"
        >>> 
        >>> # Or with default name
        >>> @timer_wrap()
        ... def another_function():
        ...     pass
    """

    def _decorator(func):
        func_name = name or func.__name__

        @wraps(func)
        def _wrapped_func(*args, **kwargs):
            """
            Wrapped function that measures execution time.

            :param args: Positional arguments passed to the original function.
            :param kwargs: Keyword arguments passed to the original function.
            :return: Return value of the original function.
            """
            timer_stack = _get_timer_stack()
            with nested_with(*map(lambda x: x.timer(func_name), timer_stack)):
                retval = func(*args, **kwargs)
            return retval

        return _wrapped_func

    return _decorator


@contextmanager
def timer(name: str):
    """
    Context manager for timing code execution using the active timer stack.

    This context manager measures the execution time of the code block within it
    and records the timing using all active TimeManager instances in the current
    timer stack. This provides a convenient way to time arbitrary code sections
    without needing direct access to a TimeManager instance.

    :param name: Name to associate with this timing measurement.
    :type name: str
    :yields: None

    Example::

        >>> with timer('data_loading'):
        ...     # Load and process data
        ...     data = load_large_dataset()
        ...     processed_data = preprocess(data)
        >>> 
        >>> with timer('model_forward'):
        ...     output = model(input_tensor)
    """
    timer_stack = _get_timer_stack()
    with nested_with(*map(lambda x: x.timer(name), timer_stack)):
        yield
