import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import continuous_armset_bai.subgrad_subcircle as subgrad_subcircle
import pickle
import scipy.stats
import math
import pytest


def to_worst_logposterior(r, t) -> float:
    mu_over_sigma = math.sqrt(-r) * t
    return scipy.stats.norm.logcdf(-mu_over_sigma)


y_name = r"$\log(p_t)$"


def add_res(d, path, method_name):
    with open(path, "rb") as fp:
        res = pickle.load(fp)
    y = y_name
    t = "time step"
    r = "r"

    for a in res:
        d[r].extend(a.rates)
        d[y].extend(to_p(a.rates))
        d[t].extend(range(len(a.rates)))

        d["method"].extend([method_name for _ in range(len(a.rates))])


def to_p(rs):
    rs = np.array(rs)
    n = rs.shape[0]
    return np.array(
        [to_worst_logposterior(r, i) for i, r in zip(np.arange(1, n + 1), rs)]
    )


def to_res_df(a=0.04, b=0.04, niter=50, fw_tol=1e-3, lamv=0, eps=None):
    y = y_name
    t = "time step"
    r = "r"
    d = {r: [], y: [], t: [], "method": []}

    mu = subgrad_subcircle.feature_vec(math.pi * a)
    bounds = np.array([(0, math.pi * b)])
    (v, f) = subgrad_subcircle.naive_opt_char_time_inv(50, mu, 1e-2, bounds)
    if eps is not None:
        add_res(
            d,
            f"results/sbg_res_p_{fw_tol:.3e}_{a:.3f}_{b:.3f}_{eps:.3f}_lamv{lamv}.pickle",
            "Ours",
        )
        add_res(d, f"results/unif_res_p_{a:.3f}_{b:.3f}_{eps:.3f}.pickle", "Uniform")
    else:
        add_res(d, f"results/sbg_res{a:.3f}_{b:.3f}_lamv{lamv}.pickle", "Ours")
    for beta in [100.0]:
        if eps is not None:
            add_res(d, f"results/non_adap_p_{a:.3f}_{b:.3f}_{eps:.3f}.pickle", f"MVR")
        else:
            add_res(d, f"results/ucb_res{a:.3f}_{b:.3f}_{beta}.pickle", f"Non-ad")

    return pd.DataFrame(d)


def load_res(a=0.2, b=1.0, niter=100, fw_tol=1e-3, lamv=0, eps=1e-2):
    with open(
        f"results/sbg_res_p_{fw_tol:.3e}_{a:.3f}_{b:.3f}_{eps:.3f}_lamv{lamv}.pickle",
        "rb",
    ) as fp:
        return pickle.load(fp)


class TestDumpImage:
    def test_dump_image(self):
        plt.rcParams.update(
            {
                "text.usetex": False,
                "figure.dpi": 300,
                "font.size": 35,
            }
        )

        a_seq = np.linspace(0.0, 1.8, 10)
        b_seq = np.linspace(0.0, 1.8, 10)[1:]
        a_b_seq_all = [(a, b) for a in a_seq for b in b_seq if a < b]

        for n in range(3):
            print(n)
            if n < 2:
                ncols = 3
                nrows = 6
            elif n == 2:
                ncols = 3
                nrows = 3
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(ncols * 8 * 1.0 + 20, nrows * 6 + 18),
            )

            ij_seq = [(i, j) for i in range(nrows) for j in range(ncols)]
            for (i, j), (a, b) in zip(ij_seq, a_b_seq_all[n * len(ij_seq) :]):
                ax = axes[i][j]
                sns.lineplot(
                    data=to_res_df(a=a, b=b, eps=1e-2, fw_tol=1e-5),
                    x="time step",
                    y=y_name,
                    hue="method",
                    ax=ax,
                )
                ax.set_xlabel("round", loc="left")
                ax.set_title(
                    rf"a, b = {a:.1f}, {b:.1f}",
                )
            plt.savefig(
                f"results/images/experiment{n}_zeta_emp.pdf",
                dpi=300,
                bbox_inches="tight",
            )
