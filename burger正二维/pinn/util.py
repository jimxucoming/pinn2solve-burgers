import sys
import logging
from typing import Callable
from functools import wraps
import time


def get_log(log_name: str = 'root') -> logging.Logger:

    new_log = logging.getLogger(log_name)
    log_format = '%(funcName)s: ''%(message)s'

    logging.basicConfig(
        stream=sys.stdout,
        format=log_format,
        level=logging.INFO
    )

    return new_log


def perf(fn: Callable):
    '''
    Performance function fn with decorators
    '''
    name = fn.__name__

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        log.info(f'started method {name}')
        ret = fn(*args, **kwargs)
        elapsed = time.time() - start
        log.info('{} took {:.4f}s'.format(name, elapsed))
        return ret
    return wrapper



log = get_log()
