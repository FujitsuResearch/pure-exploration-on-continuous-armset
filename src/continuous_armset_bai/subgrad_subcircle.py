import numpy as np
import scipy.optimize
from typing import Optional, List, Callable, Tuple, Union
from dataclasses import dataclass
from continuous_armset_bai.qfp import qfp_maximize
import math


@dataclass
class IterRes:
    xs: List[np.ndarray]
    ys: List[float]
    rates: List[float]
    vs: Optional[List[float]]


def char_time_inv_opt_fn(
    mu: np.ndarray, cov_mat: np.ndarray, x: np.ndarray, z: np.ndarray, eps: float
) -> float:
    vw_inv = np.linalg.inv(cov_mat)
    a = z - x
    return ((vw_inv @ a).dot(a)) / (eps + mu.dot(a)) ** 2


def char_time_inv_opt_fn_v(
    mu: np.ndarray,
    cov_mat: np.ndarray,
    z: np.ndarray,
    eps: float,
    bounds: np.ndarray,
    niter=50,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(0)
    vw_inv = np.linalg.inv(cov_mat)
    x0 = rng.uniform(low=bounds.T[0], high=bounds.T[1])

    def a_fn(a):
        b = z - feature_vec(a)
        return (vw_inv @ b).dot(b)

    b_fn = lambda a: (eps + mu.dot(z - feature_vec(a))) ** 2
    return qfp_maximize(a_fn, b_fn, x0, bounds, niter=niter)


def z_opt(mu, bounds):
    rng = np.random.default_rng(0)
    a0 = rng.uniform(low=bounds.T[0], high=bounds.T[1])
    z = feature_vec(
        scipy.optimize.minimize(
            lambda a: -mu.dot(feature_vec(a)), x0=a0, bounds=bounds
        ).x
    )
    return z


def naive_opt_char_time_inv(
    xnpt: int, mu: np.ndarray, eps: float, bounds: np.ndarray, niter=30
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(0)
    fin_arms = np.array(
        [
            feature_vec(a)
            for a in rng.uniform(low=bounds.T[0], high=bounds.T[1], size=(xnpt, 1))
        ]
    )
    z = z_opt(mu, bounds)

    def covar_mat_fin_arm(w, fin_arms):
        d = fin_arms.shape[1]
        res = np.zeros(shape=(d, d))
        for p, a in zip(w, fin_arms):
            res += p * np.outer(a, a)
        return res

    def f(w):
        v = covar_mat_fin_arm(w, fin_arms)
        return char_time_inv_opt_fn_v(mu, v, z, eps, bounds, niter=niter)[1]

    constraints_w = {"type": "eq", "fun": lambda x: 1 - x.sum()}
    opt_res = scipy.optimize.minimize(
        f,
        np.ones(xnpt) / xnpt,
        bounds=[(0, 1) for _ in range(xnpt)],
        constraints=constraints_w,
        method="SLSQP",
    )
    return covar_mat_fin_arm(opt_res.x, fin_arms), opt_res.fun


def feature_vec(a: Union[float, np.ndarray]) -> np.ndarray:
    if isinstance(a, np.ndarray):
        assert a.size == 1
        a = a.flatten()
        a = a[0]
    return np.array([math.cos(a), math.sin(a)])


def mu_vec(xs, ys, lam):
    v = cov_mat_emp(xs, lam)
    a = np.sum([x * y for x, y in zip(xs, ys)], axis=0)
    return np.linalg.inv(v) @ a


def cov_mat_emp(xs, lam):
    d = xs[0].size
    return np.sum([np.outer(x, x) for x in xs], axis=0) + np.eye(d) * lam


def rate_emp(xs, ys, lam, zeta: np.ndarray | None, eps: float, bounds: np.ndarray, rng):
    
    ys = np.array(ys)
    mu = mu_vec(xs, ys, lam)
    vt = cov_mat_emp(xs, lam)
    
    if zeta is None:
        zeta = feature_vec(
                scipy.optimize.minimize(
                    lambda a: -mu.dot(feature_vec(a)),
                    x0=rng.uniform(low=bounds.T[0], high=bounds.T[1]),
                    bounds=bounds,
                ).x
            )

    
    t = ys.size
    v = vt / t
    (_, fval) = char_time_inv_opt_fn_v(mu, v, zeta, eps, bounds)
    return -(fval ** (-1))


def sherman_morrison(ainv: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = ainv @ v
    return ainv - np.outer(u, u) / (1 + u.dot(v))


def cov_mat(xs, ps):
    return sum([p * np.outer(x, x) for x, p in zip(xs, ps)])


def subgrad_method(
    niter: int,
    bounds: np.ndarray,
    reward_v: np.ndarray,
    eps: float = 1e-2,
    alpha: float = 1.0,
    lam: float = 1e-2,
    lamv: float = 1e-3,
    fw_tol: float = 1e-3,
    sigma: float = 1e-2,
    seed: int = 0,
    verbose: bool = False,
) -> IterRes:
    xs = []
    ys = []
    rates = []
    vs = []
    rng = np.random.default_rng(0)
    rng_rw = np.random.default_rng(seed)
    a0 = rng.uniform(low=bounds.T[0], high=bounds.T[1])
    # z_true = feature_vec(
    #     scipy.optimize.minimize(
    #         lambda a: -reward_v.dot(feature_vec(a)), x0=a0, bounds=bounds
    #     ).x
    # )
    z_true = reward_v
    x_cands_t = None
    x_prob_t = None
    b_vec = np.zeros(2)
    vinv = np.eye(2) * lam ** (-1)
    # xs_unif = non_adaptive_pts(niter=50, bounds=bounds, lam=lam, seed=seed)
    xs_unif = [
        feature_vec(a)
        for a in rng.uniform(low=bounds.T[0], high=bounds.T[1], size=(10, 1))
    ]

    vpi_unif = np.mean([np.outer(x, x) for x in xs_unif], axis=0)
    a = None
    for t in range(1, niter + 1):
        if x_cands_t is not None:
            if rng.binomial(n=1, p=t ** (-alpha)):
                x = rng.choice(xs_unif)
            else:
                x = rng.choice(x_cands_t, p=x_prob_t)
        else:
            x = rng.choice(xs_unif)
            # x = feature_vec(rng.uniform(low=bounds.T[0], high=bounds.T[1], size=(1, 1)))
        y = reward_v.dot(x) + rng_rw.normal(loc=0, scale=sigma)
        xs.append(x)
        ys.append(y)
        rates.append(rate_emp(xs, ys, lam, zeta=None, eps=eps, bounds=bounds, rng=rng))
        if verbose:
            print(rates[-1])
        b_vec += x * y
        vinv = sherman_morrison(vinv, x)
        # mu = vinv @ b_vec
        mu = mu_vec(xs, ys, lam)
        if x_cands_t is not None:
            v1 = (
                cov_mat(x_cands_t, x_prob_t) * (1 - t ** (-alpha))
                + t ** (-alpha) * vpi_unif
            )
        else:
            v1 = vpi_unif
        vs.append(v1)
        v1inv = np.linalg.inv(v1)
        z = feature_vec(
            scipy.optimize.minimize(
                lambda a: -mu.dot(feature_vec(a)),
                x0=rng.uniform(low=bounds.T[0], high=bounds.T[1]),
                bounds=bounds,
            ).x
        )
        xi_a, f_at_v = char_time_inv_opt_fn_v(mu, v1, z, eps, bounds)
        assert xi_a >= bounds.T[0] and xi_a <= bounds.T[1]
        xi = feature_vec(xi_a)
        a = z - xi
        b = v1inv @ a
        print("b norm", np.linalg.norm(b))
        print("mu.dot(a)", mu.dot(a))
        subgrad = (eps + mu.dot(a)) ** (-2) * np.outer(b, b)
        if a is not None:
            a = subgrad.trace()
        # subgrad = subgrad / subgrad.trace()
        print("trace", subgrad.trace())

        # w_mat = (
        #     v1 + subgrad * a ** (-1) * max(math.pow(t + 1, -0.5), 0.001) - 2 * lamv * v1
        # )

        w_mat = v1 + subgrad * a ** (-1) * math.pow(t + 1, -0.5) - 2 * lamv * v1
        # print(
        #     "f at w, f at v1",
        #     char_time_inv_opt_fn_v(mu, w_mat, z, eps, bounds)[1],
        #     f_at_v,
        # )
        # print(
        #     "norm1",
        #     np.linalg.norm((eps + mu.dot(a)) ** (-2) * subgrad / math.sqrt(t + 1)),
        # )
        print("norm2", np.linalg.norm(v1))
        if x_cands_t is None:
            x_cands_t = np.array(
                [
                    feature_vec(
                        scipy.optimize.minimize(
                            lambda a: -(w_mat @ feature_vec(a)).dot(feature_vec(a)),
                            x0=rng.uniform(low=bounds.T[0], high=bounds.T[1]),
                            bounds=bounds,
                        ).x
                    )
                ]
            )
            x_prob_t = np.array([1.0])
        niter_proj = t**2
        v_tilde = cov_mat(x_cands_t, x_prob_t)

        def proj_obj_fn(v):
            return np.linalg.norm(w_mat - v) ** 2

        dist = proj_obj_fn(v_tilde)
        for s in range(3, 3 + niter_proj):
            xi = feature_vec(
                scipy.optimize.minimize(
                    lambda a: -((w_mat - v_tilde) @ feature_vec(a)).dot(feature_vec(a)),
                    x0=rng.uniform(low=bounds.T[0], high=bounds.T[1]),
                    bounds=bounds,
                ).x
            )
            xi_outer = np.outer(xi, xi)
            c = scipy.optimize.minimize(
                lambda c: proj_obj_fn((1 - c) * v_tilde + c * xi_outer),
                bounds=[(0, 1)],
                x0=1 / s,
            ).x

            # for c in cs:
            #     v_tilde_new =
            #     dists.append(proj_obj_fn(v_tilde_new))
            # c = cs[np.argmin(dists)]
            v_tilde_new = (1 - c) * v_tilde + c * np.outer(xi, xi)
            dits_new = proj_obj_fn(v_tilde_new)
            x_cands_t = np.vstack((x_cands_t, xi))
            x_prob_t = np.append((1 - c) * x_prob_t, c)
            if np.max(np.abs(dits_new - dist)) < fw_tol:
                break
            v_tilde = v_tilde_new
            dist = dits_new

    return IterRes(xs=xs, ys=ys, rates=rates, vs=vs)


def unif_method(
    niter: int,
    bounds: np.ndarray,
    reward_v: np.ndarray,
    eps: float = 1e-2,
    lam: float = 1e-2,
    sigma: float = 1e-2,
    seed: int = 0,
    verbose: bool = False,
):
    rng_rw = np.random.default_rng(seed)
    rng = np.random.default_rng(0)
    xs = []
    ys = []
    rates = []
    a0 = rng.uniform(low=bounds.T[0], high=bounds.T[1])
    # z_true = feature_vec(
    #     scipy.optimize.minimize(
    #         lambda a: -reward_v.dot(feature_vec(a)), x0=a0, bounds=bounds
    #     ).x
    # )
    z_true = reward_v
    for t in range(1, niter + 1):
        x = feature_vec(rng.uniform(low=bounds.T[0], high=bounds.T[1], size=(1, 1)))
        y = reward_v.dot(x) + rng_rw.normal(loc=0, scale=sigma)
        xs.append(x)
        ys.append(y)
        rates.append(rate_emp(xs, ys, lam, zeta=None, eps=eps, bounds=bounds, rng=rng))
        if verbose:
            print(rates[-1])
    return IterRes(xs=xs, ys=ys, rates=rates, vs=None)


def linmvr(
    niter: int,
    bounds: np.ndarray,
    reward_v: np.ndarray,
    eps: float = 1e-2,
    lam: float = 1e-2,
    sigma: float = 1e-2,
    seed: int = 0,
    verbose: bool = False,
    beta: float = 0.1,
):
    rng_rw = np.random.default_rng(seed)
    rng = np.random.default_rng(0)
    xs = []
    ys = []
    rates = []
    vs = []
    a0 = rng.uniform(low=bounds.T[0], high=bounds.T[1])
    # z_true = feature_vec(
    #     scipy.optimize.minimize(
    #         lambda a: -reward_v.dot(feature_vec(a)), x0=a0, bounds=bounds
    #     ).x
    # )
    z_true = reward_v
    b_vec = np.zeros(2)
    vinv = np.eye(2) * lam ** (-1)

    for t in range(1, niter + 1):

        mu = vinv @ b_vec

        def ucb_m(a):
            x = feature_vec(a)
            return -(math.sqrt((vinv @ x).dot(x)))

        a_rand = rng.uniform(low=bounds.T[0], high=bounds.T[1], size=(1,))

        a = scipy.optimize.minimize(ucb_m, x0=a_rand, bounds=bounds).x
        x = feature_vec(a)

        y = reward_v.dot(x) + rng_rw.normal(loc=0, scale=sigma)
        b_vec += x * y
        vinv = sherman_morrison(vinv, x)
        xs.append(x)
        ys.append(y)
        vs.append(np.outer(x, x))
        rates.append(rate_emp(xs, ys, lam, zeta=None, eps=eps, bounds=bounds, rng=rng))
        if verbose:
            print(rates[-1])
    return IterRes(xs=xs, ys=ys, rates=rates, vs=vs)


def non_adaptive_pts(
    niter: int,
    bounds: np.ndarray,
    lam: float = 1e-2,
    seed: int = 0,
):
    rng_rw = np.random.default_rng(seed)
    rng = np.random.default_rng(0)
    xs = []
    ys = []
    rates = []
    a0 = rng.uniform(low=bounds.T[0], high=bounds.T[1])
    vinv = np.eye(2) * lam ** (-1)

    for t in range(1, niter + 1):

        def ucb_m(a):
            x = feature_vec(a)
            return -(math.sqrt((vinv @ x).dot(x)))

        a_rand = rng.uniform(low=bounds.T[0], high=bounds.T[1], size=(1,))

        a = scipy.optimize.minimize(ucb_m, x0=a_rand, bounds=bounds).x
        x = feature_vec(a)

        vinv = sherman_morrison(vinv, x)
        xs.append(x)
    return xs


def nonadaptive_method(
    niter: int,
    bounds: np.ndarray,
    reward_v: np.ndarray,
    eps: float = 1e-2,
    lam: float = 1e-2,
    sigma: float = 1e-2,
    seed: int = 0,
):
    rng_rw = np.random.default_rng(seed)
    rng = np.random.default_rng(0)
    xs = []
    ys = []
    rates = []
    a0 = rng.uniform(low=bounds.T[0], high=bounds.T[1])
    vinv = np.eye(2) * lam ** (-1)
    z_true = feature_vec(
        scipy.optimize.minimize(
            lambda a: -reward_v.dot(feature_vec(a)), x0=a0, bounds=bounds
        ).x
    )
    x_cands = []
    for s in range(500):

        def mposterior_var(a):
            x = feature_vec(a)
            return -(vinv @ x).dot(x)

        opt_res = scipy.optimize.minimize(mposterior_var, x0=a0, bounds=bounds)
        print(opt_res.fun)
        x = feature_vec(opt_res.x)
        vinv = sherman_morrison(vinv, x)
        x_cands.append(x)
    for t in range(1, niter + 1):
        x = rng.choice(x_cands)
        y = reward_v.dot(x) + rng_rw.normal(loc=0, scale=sigma)
        xs.append(x)
        ys.append(y)
        rates.append(rate_emp(xs, ys, lam, zeta=z_true, eps=eps, bounds=bounds))
    return IterRes(xs=xs, ys=ys, rates=rates)
