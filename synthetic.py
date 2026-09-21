#%%
from my_core import sample_masks, design_matrix, ols_fit, realized_cest, certified_floor, certified_set, log_pk_over_delta
import numpy as np
import math
from itertools import combinations

#%%
def sample_degree1_problem(N, beta_true, intercept = 0.0, sigma_obs = 0.0, rng = None):
    d = len(beta_true)

    Z = sample_masks(N, d, rng)

    X = design_matrix(Z, 1, False)

    query_noise = sigma_obs * rng.standard_normal(N)

    y = intercept + X @ beta_true + query_noise

    return Z, y

#%%
#Test thử với sigma_obs = 0, ngân sách lớn thậm chí thừa
beta_true = [0.40, -0.30, 0.02, 0.00, 0.15, -0.01]
d = len(beta_true)
intercept = 0.70
N = 13
sigma_obs = 0

rng_clean = np.random.default_rng(42)
Z_clean, y_clean = sample_degree1_problem(N, beta_true, intercept, sigma_obs, rng_clean)

beta_hat, intercept_hat, _ = ols_fit(Z_clean, y_clean, 1)

full_error = np.max(
        np.abs(np.concatenate(([intercept_hat], beta_hat))
               - np.concatenate(([intercept], beta_true)))
    )

print("=== Exact recovery ===")
print("intercept true:", intercept)
print("intercept hat :", intercept_hat)
print("beta true     :", beta_true)
print("beta hat      :", beta_hat)
print("max error     :", full_error)

# %% sigma_obs > 0
sigma_obs = 0.05
rng_noisy = np.random.default_rng(123)

Z, y =sample_degree1_problem(N=N, beta_true=beta_true, intercept=intercept, sigma_obs= sigma_obs, rng=rng_noisy)

beta_hat2, intercept_hat2, _ = ols_fit(Z, y, K=1)

c_est = realized_cest(Z, K=1)

floor = certified_floor(c_est=c_est, sigma_obs=sigma_obs, m_ub=0.0, B_pop_ub=0.0, d=d, N=N, K=1)

certified = certified_set(beta_hat2, floor)

print("\n=== Noisy certificate smoke test ===")
print("C_est:", c_est)
print("floor:", floor)
print()
print(" idx | beta_true | beta_hat | certified | sign_hat | sign_true")

for i in range(d):
    print(
        f"{i:>4} | "
        f"{beta_true[i]:>9.4f} | "
        f"{beta_hat2[i]:>8.4f} | "
        f"{str(bool(certified[i])):>9} | "
        f"{np.sign(beta_hat2[i]):>8.0f} | "
        f"{np.sign(beta_true[i]):>9.0f}"
    )

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
def emperical_leakage(Z, r):
    Z = np.asarray(Z, dtype=float)
    r = np.asarray(r, dtype=float)
    N = Z.shape[0]
    X_effect = design_matrix(Z, K = 1, intercept=False)
    X_aug = design_matrix(Z, K=1, intercept= True)

    pre_eta = (1/N) * (X_aug.T @ r)

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
    eta_cube = emperical_leakage(Z_cube, r_cube)

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

            eta_values.append(emperical_leakage(Z, r))

        print(
            f"{N:>3} | "
            f"{np.mean(eta_values):>10.6f} | "
            f"{np.std(eta_values):>9.6f}"
        )

# %% Ước lượng Cm
def calibrate_leakage(
    d=30,
    n_active=4,
    n_trials=40,
    m_grid=(0.005, 0.02, 0.05, 0.1, 0.2),
    N_grid=(250, 500, 1000, 2000, 4000),
):

    K = 1
    L = log_pk_over_delta(d, K, delta = None, split=3)
    results = []

    print("=== Leakage calibration ===")
    print(f"d={d}, p1={d + 1}, L={L:.6f}")
    print("m_resid |    N | mean_eta | predicted_scale | C_m_emp")
    for m_resid in m_grid:
        for N in N_grid:
            eta_values = []

            for trial in range(n_trials):
                _, _, _, resid_fn, _ = make_synthetic_function(d, n_active, beta_active = 0.0, m_resid=m_resid, seed=7 * trial + N)
                rng = np.random.default_rng(
                    123 + trial + N
                )

                Z = sample_masks(N, d, rng)
                r = resid_fn(Z)
                eta_values.append(
                    emperical_leakage(Z, r)
                )
            mean_eta = float(np.mean(eta_values))
            std_eta = float(np.std(eta_values))
            predicted_scale = math.sqrt(m_resid * L / N)
            c_m_emp = mean_eta / predicted_scale

            row = {
                "m_resid": m_resid,
                "N": N,
                "mean_eta": mean_eta,
                "std_eta": std_eta,
                "predicted_scale": predicted_scale,
                "c_m_emp": c_m_emp,
            }

            results.append(row)

            print(
                f"{m_resid:>7.3f} | "
                f"{N:>4d} | "
                f"{mean_eta:>8.6f} | "
                f"{predicted_scale:>15.6f} | "
                f"{c_m_emp:>7.3f}"
            )
    return results

if __name__ == "__main__":
    rows = calibrate_leakage()
# %%
