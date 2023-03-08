from time import time
from datetime import timedelta
from functools import wraps
from loguru import logger
import sys

logger.remove(0)
logger.add(sys.stderr,
           format="<g>{time:YYYY-MM-DD HH:mm:ss.SS!UTC}</g> <r>|</r> <y>{level}</y> <r>|</r> <w>{message}</w>",
           level="INFO",
           colorize=True)


def timeit_log(func):
    """
    Time estimation wrapper
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.opt(colors=True).info(f"<le>=== Run method {func.__name__} ===</le>")
        start = time()
        ret = func(*args, **kwargs)
        end = time()
        logger.opt(colors=True).info(f'<m>Method {func.__name__} is finished! '
                                     f'Processing time: {str(timedelta(seconds = end - start))}</m>')
        return ret
    return wrapper


def trace_log(func):
    """
    Time-logging wrapper
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Run method {func.__name__}...")
        ret = func(*args, **kwargs)
        logger.info(f"Method {func.__name__} has been finished successfully!")
        return ret
    return wrapper
