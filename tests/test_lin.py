from continuous_armset_bai.qfp import qfp_maximize
import continuous_armset_bai.subgrad_subcircle as subgrad_subcircle
import numpy as np
import math
import pickle

import pytest


class TestLinModel:
    @pytest.mark.skip(reason="done")
    def test_qfp(self):
        rng = np.random.default_rng(0)
        dim = 4
        v1 = rng.standard_normal(size=dim)
        v2 = rng.standard_normal(size=dim)
        v2 = v2 / np.linalg.norm(v2)
        v = rng.standard_normal(size=(dim, dim))
        v = v @ v.T
        a_fn = lambda x: ((x - v1) @ v).dot(x - v1)
        b_fn = lambda x: (1.0 + v2.dot(x)) ** 2
        bounds = [(0, 1) for _ in range(dim)]
        bounds_a = np.array(bounds)
        xs = rng.uniform(low=bounds_a.T[0], high=bounds_a.T[1], size=(100, dim))
        x0 = xs[0]
        (x, val) = qfp_maximize(a_fn, b_fn, x0=x0, bounds=bounds, niter=100)
        fs = [a_fn(x) / b_fn(x) for x in xs]
        print("qfp:", val, f"unif: {np.max(fs)}")
        assert val > np.max(fs)

    @pytest.mark.skip(reason="done")
    def test_qfp_chartime(self):
        rng = np.random.default_rng(0)
        mu = subgrad_subcircle.feature_vec(math.pi / 4)
        bounds = np.array([(0, math.pi / 3)])
        cov_mat = sum(
            [
                np.outer(
                    subgrad_subcircle.feature_vec(a), subgrad_subcircle.feature_vec(a)
                )
                for a in [0, math.pi / 3]
            ]
        )
        z = mu
        eps = 1e-3
        (x, fval) = subgrad_subcircle.char_time_inv_opt_fn_v(
            mu, cov_mat, z, eps=eps, bounds=bounds
        )
        xs = [
            subgrad_subcircle.feature_vec(a)
            for a in rng.uniform(low=bounds.T[0], high=bounds.T[1], size=(100, 1))
        ]
        fs = [
            subgrad_subcircle.char_time_inv_opt_fn(mu, cov_mat, x, z, eps) for x in xs
        ]
        print(f"qfp-val: {fval:.2e}", f"rand-search: {np.max(fs):.2e}")
        assert fval > np.max(fs)

    def test_run_expriments(self):
        eps = 1e-2
        for fw_tol in [1e-5]:
            for a in np.linspace(0.0, 1.8, 10):
                for b in np.linspace(0.0, 1.8, 10)[1:]:
                    if a < b:
                        # try:
                        mu = subgrad_subcircle.feature_vec(math.pi * a)
                        bounds = np.array([(0, math.pi * b)])
                        niter = 100
                        repeat = 10
                        lamv = 0
                        sbg_res = [
                            subgrad_subcircle.subgrad_method(
                                niter,
                                alpha=1,
                                bounds=bounds,
                                reward_v=mu,
                                eps=eps,
                                fw_tol=fw_tol,
                                seed=i,
                                verbose=True,
                                lamv=lamv,
                                sigma=1e-2,
                            )
                            for i in range(repeat)
                        ]
                        with open(
                            f"results/sbg_res_p_{fw_tol:.3e}_{a:.3f}_{b:.3f}_{eps:.3f}_lamv{lamv}.pickle",
                            "wb",
                        ) as fp:
                            pickle.dump(sbg_res, fp)

                        unif_res = [
                            subgrad_subcircle.unif_method(
                                niter, bounds=bounds, reward_v=mu, eps=eps, seed=i
                            )
                            for i in range(repeat)
                        ]
                        with open(
                            f"results/unif_res_p_{a:.3f}_{b:.3f}_{eps:.3f}.pickle",
                            "wb",
                        ) as fp:
                            pickle.dump(unif_res, fp)

                        ucb_res = [
                            subgrad_subcircle.linmvr(
                                niter,
                                bounds=bounds,
                                reward_v=mu,
                                eps=eps,
                                seed=i,
                                sigma=1e-2,
                            )
                            for i in range(repeat)
                        ]
                        with open(
                            f"results/non_adap_p_{a:.3f}_{b:.3f}_{eps:.3f}.pickle",
                            "wb",
                        ) as fp:
                            pickle.dump(ucb_res, fp)
                    # except:
                    #     pass