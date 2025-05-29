import numpy as np
import scipy.optimize
from typing import Tuple, Callable, List


def qfp_maximize(
    a_fn: Callable[[np.ndarray], float],
    b_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    niter: int,
    q_tol: float = 1e-5,
    verbose=False,
) -> Tuple[np.ndarray, float]:
    q = a_fn(x0) / b_fn(x0)
    x_init = x0
    opt_res = None
    for _ in range(niter):
        opt_res = scipy.optimize.minimize(
            lambda x: -(a_fn(x) - q * b_fn(x)), x0=x_init, bounds=bounds
        )
        q_new = a_fn(opt_res.x) / b_fn(opt_res.x)
        qerr = abs(q_new - q)
        if verbose:
            print(f"qerr: {qerr}")
        if qerr < q_tol:
            break
        q = q_new
        x_init = opt_res.x
    return (opt_res.x, a_fn(opt_res.x) / b_fn(opt_res.x))
