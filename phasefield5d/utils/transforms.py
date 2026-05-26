import numpy as np


def safe_log10(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    m = x > 0
    out[m] = np.log10(x[m])
    return out


def K_to_C(K):
    return K - 273.0


def C_to_K(C):
    return C + 273.0


def reciprocal(x):
    x = np.array(x, float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1.0 / x[~near_zero]
    return x


def reciprocal_times_2_pi(x):
    x = np.array(x, float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 2.0 * np.pi / x[~near_zero]
    return x
