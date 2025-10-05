from functools import wraps
from typing import Optional

from hbutils.reflection import nested_with

from .manage import _get_timer_stack


def timer_wrap(name: Optional[str] = None):
    def _decorator(func):
        func_name = name or func.__name__

        @wraps(func)
        def _wrapped_func(*args, **kwargs):
            timer_stack = _get_timer_stack()
            with nested_with(*map(lambda x: x.timer(func_name), timer_stack)):
                retval = func(*args, **kwargs)
            return retval

        return _wrapped_func

    return _decorator
