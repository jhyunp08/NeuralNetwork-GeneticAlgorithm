from functools import wraps, cache, lru_cache
import time
import numpy as np

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

def HSL2RGB(H, S, L, asHex=True):
    """
    H: hue in [0, 360]
    S: saturation in [0, 1]
    L: lightness in [0, 1]
    """

    C = (1 - abs(2*L-1)) * S
    H1 = H / 60
    X = C * (1 - abs((H1 % 2)-1))
    index_conv = [(1, 2, 0), (2, 1, 0), (0, 1, 2), (0, 2, 1), (2, 0, 1), (1, 0, 2), (1, 2, 0)]
    inds = index_conv[int(H1)]
    vars = (0, C, X)
    R1, G1, B1 = (vars[inds[0]], vars[inds[1]], vars[inds[2]])
    m = L - C/2

    R, G, B = (R1 + m , G1 + m, B1 + m)
    if asHex:
        R = int(np.round(R * 255))
        G = int(np.round(G * 255))
        B = int(np.round(B * 255))
        return f'#{format(R, "02x")}{format(G, "02x")}{format(B, "02x")}'
    else:
        return (R, G, B)
