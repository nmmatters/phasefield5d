import numpy as np
from itertools import product


def fluctuations(composition, delta=0.01, decimals=2, sort_output=True):
    composition = np.asarray(composition, dtype=float)
    steps = (-delta, 0.0, +delta)
    out = set()

    for d in product(steps, repeat=4):
        s = sum(d)
        if s not in (-delta, 0.0, +delta):
            continue
        new4 = composition + np.array(d)
        if np.any(new4 < 0) or np.any(new4 > 1):
            continue
        new0 = 1.0 - new4.sum()
        if new0 < 0 or new0 > 1:
            continue
        tup = tuple(np.round(new4, decimals=decimals).tolist())
        out.add(tup)

    res = list(out)
    if sort_output:
        res.sort()
    return res


def get_ave_err_stdv_max_min(values):
    arr = np.array(values, dtype=float)
    ave = np.mean(arr)
    stdv = np.std(arr)
    err = stdv / np.sqrt(arr.shape[0])
    return ave, err, stdv, np.max(arr), np.min(arr)


def round_to_first_nonzero(x):
    if x == 0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(x))))
    return round(x, -exponent)


def round_down_to_first_nonzero(x):
    if x == 0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(x))))
    factor = 10 ** exponent
    mantissa_down = np.floor(abs(x) / factor)
    return np.sign(x) * mantissa_down * factor
