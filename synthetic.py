#%%
from my_core import sample_masks, design_matrix, ols_fit, realized_cest, certified_floor, certified_set, log_pk_over_delta
import numpy as np
import math

#%%
def sample_degree1_problem(N, beta_true, intercept = 0.0, sigma_obs = 0.0, rng = None):
    d = len(beta_true)

    Z = sample_masks(N, d, rng)

    X = design_matrix(Z, 1, False)

    query_noise = sigma_obs * rng.standard_normal(N)

    y = intercept + X @ beta_true + query_noise

    return Z, y
# %%
"""
tạm thời đang để intercept = 0
"""
def make_synthetic_function(
    d,
    n_active, #Lượng thuộc tính thực sự đóng góp
    beta_active,
    m_resid,
    seed,
    n_hi = 200
):
    rng = np.random.default_rng(seed)

    active_idx = rng.choice(d, size=n_active, replace=False)
    active_signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=n_active
    )
    beta_true = np.zeros(d, dtype=float)
    beta_true[active_idx] = active_signs * beta_active
    active_set = set(int(i) for i in active_idx)

    #Tạo high-order mismatch (d = 6 thì tạo full high order residual)
    #hi_sets = [
    #    c
    #    for k in range(2, d + 1)
    #    for c in combinations(range(d), k)
    #]

    #Tạo hi_sets
    units = list(range(d))
    max_hi = math.comb(d, 2) + math.comb(d, 3)
    n_hi = min(n_hi, max_hi)

    hi_sets = []
    seen = set()

    while len(hi_sets) < n_hi:
        k = min(int(rng.integers(2, 4)), d)
        S = tuple(sorted(rng.choice(units, size = k, replace = False)))
        if S not in seen:
            seen.add(S)
            hi_sets.append(S)

    n_term = len(hi_sets)
    magnitude = math.sqrt(m_resid / n_term)
    #\(|\beta_{S,\rho}|=\texttt{magnitude}.\), nghĩa là betahigh thằng nào cũng như nhau và bằng magnitude
    hi_signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=n_hi
    )

    beta_hi = magnitude * hi_signs

    def g_fn(Z):
        X_effct = design_matrix(Z, K=1, intercept=False)
        y_clean =  X_effct @ beta_true + resid_fn(Z)
        return y_clean

    def chi(Z, S):
        Z = np.asanyarray(Z, dtype=float)
        result = np.ones(Z.shape[0], dtype= float)
        for i in S:
            result = result * (2 * Z[:, i] - 1)
        return result

    def resid_fn(Z):
        Z = np.asanyarray(Z, dtype=float)
        result = np.zeros(Z.shape[0], dtype= float)

        for S, coefficient in zip(hi_sets, beta_hi):
            result += coefficient * chi(Z, S)
        return result

    def sample_fn(N, sigma_obs, rng):
        Z = sample_masks(N, d, rng)
        y_clean = g_fn(Z)
        y = y_clean + sigma_obs * rng.standard_normal(N) 
        return Z, y

    return beta_true, active_set, sample_fn, resid_fn, g_fn

#%% eta_N
def empirical_leakage(Z, r):
    Z = np.asarray(Z, dtype=float)
    r = np.asarray(r, dtype=float)
    N = Z.shape[0]
    X_effect = design_matrix(Z, K = 1, intercept=False)

    pre_eta = (1/N) * (X_effect.T @ r)

    eta = float(np.max(np.abs(pre_eta)))
    return eta

# %%
"""
Test trên population
"""
from itertools import product
def test_empirical_leakage():
    d = 6
    m_resid = 0.05

    beta_true, active_set, sample_fn, resid_fn, g_fn = (
        make_synthetic_function(
            d=d,
            n_active=2,
            beta_active=0.12,
            m_resid=m_resid,
            seed=42,
        )
    )

    Z_cube = np.array(
        list(product([0.0, 1.0], repeat=d)),
        dtype=float,
    )

    r_cube = resid_fn(Z_cube)
    eta_cube = empirical_leakage(Z_cube, r_cube)

    print("=== Full cube ===")
    print("N:", len(Z_cube))              # 64
    print("m empirical:", np.mean(r_cube ** 2))
    print("eta full cube:", eta_cube)     # gần 0

    # ----- Finite random mask banks -----
    print("\n=== Finite random banks ===")
    print("  N | mean eta_N | std eta_N")

    for N in [16, 32, 64, 128, 256]:
        eta_values = []

        for trial in range(100):
            rng = np.random.default_rng(1000 + trial)

            Z = sample_masks(N, d, rng)
            r = resid_fn(Z)

            eta_values.append(empirical_leakage(Z, r))

        print(
            f"{N:>3} | "
            f"{np.mean(eta_values):>10.6f} | "
            f"{np.std(eta_values):>9.6f}"
        )


#%%
"""
Thêm hai helper nhỏ
"""
def coefficient_of_variation(values):
    values = np.asarray(values, dtype=float)

    if len(values) < 2 or np.isclose(values.mean(), 0.0):
        return float("nan")

    return float(values.std() / abs(values.mean()))

def bernstein_diagnostics(m_resid, B_sample, d, N, K=1):
    L = log_pk_over_delta(d, K, delta=None, split=3)

    sub_gaussian = math.sqrt(2.0 * m_resid * L / N)
    sub_exponential = (2.0 / 3.0) * B_sample * L / N

    if m_resid <= 0.0:
        N_dom = float("inf")
    else:
        N_dom = (2.0 * B_sample**2 / (9.0 * m_resid)) * L

    return sub_gaussian, sub_exponential, N_dom
# %% Ước lượng Cm
def calibrate_leakage(
    d=30,
    n_active=4,
    n_trials=40,
    m_grid=(0.005, 0.02, 0.05, 0.1, 0.2),
    N_grid=(250, 500, 1000, 2000, 4000),
    N_freeze=2000,
):
    K = 1
    L = log_pk_over_delta(d, K, delta=None, split=3)

    ratios_all = []
    ratios_large = []
    results = []
    not_dominated = 0

    print("=== Leakage calibration ===")
    print(f"d={d}, p1={d + 1}, L={L:.6f}")
    print(
        "m_resid |    N | mean_eta | predicted | C_m_emp | "
        "subexp | N_dom | dom?"
    )

    for m_resid in m_grid:
        for N in N_grid:
            eta_values = []
            B_samples = []

            for trial in range(n_trials):
                _, _, _, resid_fn, _ = make_synthetic_function(
                    d=d,
                    n_active=n_active,
                    beta_active=0.0,
                    m_resid=m_resid,
                    seed=7 * trial + N,
                )

                rng = np.random.default_rng(123 + trial + N)
                Z = sample_masks(N, d, rng)
                r = resid_fn(Z)

                eta_values.append(empirical_leakage(Z, r))
                B_samples.append(float(np.max(np.abs(r))))

            mean_eta = float(np.mean(eta_values))
            std_eta = float(np.std(eta_values))
            B_sample = float(np.mean(B_samples))

            predicted_scale = math.sqrt(m_resid * L / N)
            c_m_emp = mean_eta / predicted_scale

            sub_g, sub_e, N_dom = bernstein_diagnostics(
                m_resid, B_sample, d, N, K
            )
            dominated = N >= N_dom

            ratios_all.append(c_m_emp)

            if N >= N_freeze:
                ratios_large.append(c_m_emp)

            if not dominated:
                not_dominated += 1

            results.append({
                "m_resid": m_resid,
                "N": N,
                "mean_eta": mean_eta,
                "std_eta": std_eta,
                "B_sample": B_sample,
                "predicted_scale": predicted_scale,
                "c_m_emp": c_m_emp,
                "sub_gaussian": sub_g,
                "sub_exponential": sub_e,
                "N_dom": N_dom,
                "dominated": dominated,
            })

            print(
                f"{m_resid:>7.3f} | {N:>4d} | "
                f"{mean_eta:>8.6f} | {predicted_scale:>9.6f} | "
                f"{c_m_emp:>7.3f} | {sub_e:>6.4f} | "
                f"{N_dom:>5.0f} | {'yes' if dominated else 'NO'}"
            )

    C_m_all = float(np.mean(ratios_all))
    C_m_frozen = float(np.mean(ratios_large))

    print()
    print(f"C_m all-N               = {C_m_all:.3f}")
    print(
        f"C_m N >= {N_freeze} (FROZEN) = {C_m_frozen:.3f}; "
        f"CoV = {coefficient_of_variation(ratios_large):.3f}"
    )
    print(
        f"Dominated cells: "
        f"{len(results) - not_dominated}/{len(results)}"
    )

    return C_m_frozen, results

if __name__ == "__main__":
    rows = calibrate_leakage()
# %%
