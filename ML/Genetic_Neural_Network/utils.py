from functools import wraps, cache, lru_cache
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}({str([str(arg) for arg in args])[1:-1]} {str(kwargs)[1:-1]}) Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
