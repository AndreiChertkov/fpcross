import numpy as np
import teneva


def _renormalize_new(Y, a, b, n):
    a, b, n = teneva.grid_prep_opts(a, b, n)

    c = np.array([1.])
    for ci, a, b, n in zip(Y, a, b, n):
        c = c @ ci.sum(axis=1)
        c *= (b - a) / n

    s = c.item()
    return teneva.mul(1./s, Y)
