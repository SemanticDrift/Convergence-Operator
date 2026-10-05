"""Replication script for the star Convergence Operator (v2).
Requires: numpy, scipy, sympy. Run: python star_replication.py
"""
import time
from fractions import Fraction
import numpy as np
from sympy import factorint

# ---------- 1. Prime normalization f_P (sparse exponent vectors) ----------
def f_P(q):
    q = Fraction(q)
    v = {}
    for p, e in factorint(q.numerator).items():   v[p] = v.get(p, 0) + e
    for p, e in factorint(q.denominator).items(): v[p] = v.get(p, 0) - e
    return {p: e for p, e in v.items() if e != 0}

def vec(q, primes):
    v = f_P(q)
    return np.array([v.get(p, 0) for p in primes], dtype=float)

def positional(int_digits, frac_digits, base):
    n = Fraction(0)
    for d in int_digits:  n = n * base + d
    for i, d in enumerate(frac_digits, 1): n += Fraction(d, base ** i)
    return n

# ---------- 2. The star operator (affine) with spectral audit ----------
def star_affine(psi, delta, b, tol=1e-10):
    delta = np.atleast_2d(np.asarray(delta, float))
    x = psi - np.linalg.pinv(delta, rcond=tol) @ (delta @ psi - b)
    lam = np.linalg.eigvalsh(delta.T @ delta)
    k = int(np.sum(lam <= tol * lam[-1]))
    return x, dict(feas_residual=float(np.linalg.norm(delta @ x - b)),
                   kernel_dim=k, gap=float(lam[k]) if k < len(lam) else np.inf,
                   artifact=float(np.linalg.norm(psi - x)),
                   per_constraint=delta @ psi - b)

def flow(psi, delta, b, t):
    x, _ = star_affine(psi, delta, b)
    lam, V = np.linalg.eigh(np.atleast_2d(delta).T @ np.atleast_2d(delta))
    return x + V @ (np.exp(-t * lam) * (V.T @ (psi - x)))

# ---------- Experiment A: notation collapse ----------
def experiment_A():
    reps = {
        "decimal 0.5":        positional([0], [5], 10),
        "binary 0.1":         positional([0], [1], 2),
        "sexagesimal 0;30":   positional([0], [30], 60),
        "Egyptian 1/2":       Fraction(1, 2),
    }
    pts = {name: tuple(sorted(f_P(q).items())) for name, q in reps.items()}
    assert len(set(pts.values())) == 1
    print("A1 four notations of 1/2 ->", set(pts.values()))
    third = {"base-3 0.1": positional([0], [1], 3),
             "sexagesimal 0;20": positional([0], [20], 60),
             "Egyptian 1/3": Fraction(1, 3)}
    assert len({tuple(sorted(f_P(q).items())) for q in third.values()}) == 1
    trunc = positional([0], [3] * 6, 10)               # 0.333333
    gap = f_P(trunc / Fraction(1, 3))
    print("A2 0.333333 vs 1/3 gap vector:", gap,
          " relative error:", float(trunc * 3 - 1))
    assert gap != {}

# ---------- Experiment B: unit conversion, fault localization ----------
def experiment_B():
    P = (2, 3, 5, 7)
    r1 = vec(480, P)                                   # Hz
    r2 = vec(Fraction(480, 1000), P)                   # kHz
    r3 = vec(480 * 60, P)                              # per minute
    r3_bad = vec(Fraction(480 * 60 * 3, 2), P)         # mis-conversion by 3/2
    psi = np.concatenate([r1, r2, r3_bad])
    I, Z = np.eye(4), np.zeros((4, 4))
    delta = np.block([[-I, I, Z], [-I, Z, I], [Z, -I, I]])
    b = np.concatenate([-vec(1000, P), vec(60, P), vec(60, P) + vec(1000, P)])
    x, a = star_affine(psi, delta, b)
    res = a["per_constraint"].reshape(3, 4)
    print("B per-constraint residual norms [r2-r1, r3-r1, r3-r2]:",
          np.linalg.norm(res, axis=1).round(4))
    print("B kernel dim:", a["kernel_dim"], " gap:", round(a["gap"], 4),
          " post-star feasibility residual:", round(a["feas_residual"], 12))
    moved = np.linalg.norm((x - psi).reshape(3, 4), axis=1)
    print("B correction per reading [r1, r2, r3]:", moved.round(4))
    assert np.linalg.norm(res[0]) < 1e-12 and np.linalg.norm(res[1]) > 0.5
    assert a["feas_residual"] < 1e-9

# ---------- Experiment C: float 0.1 against the intended 1/10 ----------
def experiment_C():
    gap = f_P(Fraction(0.1) / Fraction(1, 10))
    small = {p: e for p, e in gap.items() if p <= 7}
    rel = float(Fraction(0.1) * 10 - 1)
    print("C float(0.1)/ (1/10) gap, primes <= 7:", small,
          " other primes:", len(gap) - len(small))
    print("C relative error:", rel, " below 2^-53:", abs(rel) < 2.0 ** -53)
    assert abs(rel) < 2.0 ** -53

# ---------- Experiment D: twelve fifths close (equal temperament) ----------
def experiment_D():
    fifths = np.full(12, 1200 * np.log2(1.5))
    x, a = star_affine(fifths, np.ones((1, 12)), np.array([8400.0]))
    print("D fifth", fifths[0].round(4), "->", x[0].round(4),
          " sum:", round(x.sum(), 9), " comma:", a["per_constraint"].round(4),
          " gap:", a["gap"])
    err = np.linalg.norm(flow(fifths, np.ones((1, 12)), np.array([8400.0]), 0.2) - x)
    print("D flow error at t=0.2:", round(err, 8),
          " bound:", round(float(np.exp(-0.2 * a["gap"]) * a["artifact"]), 8))
    assert abs(x.sum() - 8400) < 1e-9 and err <= np.exp(-0.2 * a["gap"]) * a["artifact"] + 1e-12

# ---------- Experiment G: equivalence classes of star ----------
def experiment_G():
    rng = np.random.default_rng(7)
    n, m, r = 8, 5, 4
    d = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))   # rank 4, redundant rows
    U, s, Vt = np.linalg.svd(d); rk = int((s > 1e-10 * s[0]).sum())
    ker, row = Vt[rk:].T, Vt[:rk].T
    star = lambda x, b: x - np.linalg.pinv(d) @ (d @ x - b)
    for label, b in (("feasible", d @ rng.standard_normal(n)), ("infeasible", rng.standard_normal(m))):
        idem = same = kept = res = 0.0
        for _ in range(1000):
            x = rng.standard_normal(n)
            idem = max(idem, np.linalg.norm(star(star(x, b), b) - star(x, b)))
            y = x + row @ rng.standard_normal(rk)
            same = max(same, np.linalg.norm(star(x, b) - star(y, b)))
            z = ker @ rng.standard_normal(n - rk)
            kept = max(kept, abs(np.linalg.norm(star(x, b) - star(x + z, b)) - np.linalg.norm(z)))
            if label == "feasible": res = max(res, np.linalg.norm(d @ star(x, b) - b))
        print(f"G {label}: idempotency {idem:.1e}, range(d^T) difference -> equal images {same:.1e}, "
              f"ker(d) difference preserved to {kept:.1e}, residual {res:.1e}")
        assert max(idem, same, kept, res) < 1e-10
    zero = np.zeros((2, n)); x0 = rng.standard_normal(n)          # no constraints at all
    xz, az = star_affine(x0, zero, np.zeros(2))
    print(f"G zero constraints: state unchanged ({np.linalg.norm(xz - x0):.1e}), "
          f"kernel dimension {az['kernel_dim']} of {n}, gap {az['gap']}")
    assert np.allclose(xz, x0) and az["kernel_dim"] == n and az["gap"] == np.inf

# ---------- Experiment F: mixed (nonlinear) constraint a*b + c = d ----------
def star_flow(x0, g, J, b, T=40.0):
    from scipy.integrate import solve_ivp
    f = lambda t, x: -(np.linalg.pinv(J(x)) @ (g(x) - b))
    return solve_ivp(f, (0, T), x0, rtol=1e-11, atol=1e-13, dense_output=True)

def experiment_F():
    g = lambda x: np.array([x[0] * x[1] + x[2] - x[3]])
    J = lambda x: np.array([[x[1], x[0], 1.0, -1.0]])
    b = np.zeros(1)
    for x0 in ([2, 3, 1, 5.], [-4, 7, 3, 0.], [100, -50, 9, 9.]):
        x0 = np.array(x0); sol = star_flow(x0, g, J, b); xe = sol.y[:, -1]
        r0 = abs(g(x0)[0]); bound = r0 / np.sqrt(2)
        ratio = g(sol.sol(3.0))[0] / g(x0)[0]
        print(f"F x0={x0} -> g(x_inf)={g(xe)[0]:.1e} g(3)/g(0)={ratio:.6f} "
              f"(exp(-3)={np.exp(-3):.6f}) displacement={np.linalg.norm(xe-x0):.3f} <= {bound:.3f}")
        assert abs(g(xe)[0]) < 1e-8 and abs(ratio - np.exp(-3)) < 1e-6
        assert np.linalg.norm(xe - x0) <= bound + 1e-9

# ---------- Experiments H, I, J, K: error reduction when the true state is consistent ----------
def _star(psi, d, b): return psi - np.linalg.pinv(d, rcond=1e-10) @ (d @ psi - b)

def _incidence(n, edges):
    B = np.zeros((n, len(edges)))
    for k, (u, v) in enumerate(edges): B[u, k], B[v, k] = -1.0, 1.0
    return B

def _variance_law(d, b, x_true, rng, sigma=1.0, trials=20000):
    n = len(x_true); raw = ref = 0.0; never_worse = True
    for _ in range(trials):
        eps = sigma * rng.standard_normal(n); s = _star(x_true + eps, d, b)
        raw += eps @ eps; ref += np.sum((s - x_true) ** 2)
        never_worse &= bool(np.linalg.norm(s - x_true) <= np.linalg.norm(eps) + 1e-12)
    return raw / trials, ref / trials, never_worse

def experiment_H():
    rng = np.random.default_rng(2026)
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3),(1,4),(2,5),(0,2)]
    B = _incidence(6, edges); x_true = 5 * rng.standard_normal(10); s = B @ x_true
    law = 10 - np.linalg.matrix_rank(B)
    raw, ref, nw = _variance_law(B, s, x_true, rng)
    print(f"H flow network: MSE {raw:.3f} -> {ref:.3f} (law {law}), never worse: {nw}")
    assert nw and abs(ref - law) / law < 0.03
    hits, T = 0, 2000
    for _ in range(T):
        psi = x_true + rng.standard_normal(10); k = rng.integers(10); psi[k] += 8.0
        hits += set(np.argsort(np.abs(B @ psi - s))[-2:]) == set(edges[k])
    print(f"H gross error 8 sigma: endpoints are the two largest node residuals in {100*hits/T:.1f}% of {T} trials")

def experiment_I():
    rng = np.random.default_rng(2027)
    d = np.array([[1,-1,-1,0,0,0,0,0],[0,1,0,-1,-1,0,0,0],[0,0,1,0,0,-1,-1,-1]], float)
    bottom = rng.uniform(50, 150, 5); A, Bv = bottom[:2].sum(), bottom[2:].sum()
    x_true = np.concatenate([[A + Bv, A, Bv], bottom]); law = 8 - np.linalg.matrix_rank(d)
    raw, ref, nw = _variance_law(d, np.zeros(3), x_true, rng)
    print(f"I hierarchical forecasts: MSE {raw:.3f} -> {ref:.3f} (law {law}), never worse: {nw}")
    assert nw and abs(ref - law) / law < 0.03

def experiment_J():
    rng = np.random.default_rng(2028)
    edges = [(i, (i + 1) % 8) for i in range(8)] + [(0,4),(1,5),(2,6),(3,7)]
    B = _incidence(8, edges); _, S, Vt = np.linalg.svd(B); C = Vt[int((S > 1e-10).sum()):]
    x_true = B.T @ (10 * rng.standard_normal(8)); law = 12 - len(C)
    raw, ref, nw = _variance_law(C, np.zeros(len(C)), x_true, rng)
    print(f"J clock offsets (loop closure): MSE {raw:.3f} -> {ref:.3f} (law {law}), never worse: {nw}")
    assert nw and abs(ref - law) / law < 0.03

def experiment_K():
    x_true = np.array([1.0, 2.0])
    err = np.linalg.norm(_star(x_true, np.array([[1.0, -1.0]]), np.zeros(1)) - x_true)
    print(f"K false constraint x0 = x1 moves a noiseless true state by {err:.4f}")
    assert abs(err - 2 ** -0.5) < 1e-12

# ---------- Experiment E: cost scaling ----------
def experiment_E():
    rng = np.random.default_rng(0)
    for n in (100, 200, 400, 800):
        d = rng.standard_normal((n // 2, n)); b = rng.standard_normal(n // 2)
        psi = rng.standard_normal(n)
        t0 = time.perf_counter(); x, a = star_affine(psi, d, b)
        print(f"E n={n:4d} constraints={n//2:4d} time={time.perf_counter()-t0:7.3f}s "
              f"feasibility residual={a['feas_residual']:.2e}")

if __name__ == "__main__":
    for e in (experiment_A, experiment_B, experiment_C, experiment_D, experiment_G, experiment_F, experiment_H, experiment_I, experiment_J, experiment_K, experiment_E):
        e()
    print("ALL CHECKS PASSED")
