import math
from itertools import combinations
import numpy as np
def p_K(d: int, K: int) -> int:
    """Trả về số cột (tính cả intercept) theo bậc tương tác K.

    - K = 1: p_1 = 1 + d
    - K = 2: p_2 = 1 + d + C(d, 2)
    """
    if d < 0:
        raise ValueError("d phải là số nguyên không âm.")

    if K == 1:
        return 1 + d
    elif K == 2:
        return 1 + d + math.comb(d, 2)
    else:
        raise ValueError(f"Chỉ hỗ trợ K in {{1, 2}}, nhận được: K = {K}")

def feature_subsets(d, K):
    """Trả về danh sách các tập chỉ số đặc trưng (không gồm intercept).

    - K = 1: [(0,), (1,), ..., (d-1,)]
    - K = 2: các singleton trước, sau đó là mọi cặp (i, j) với i < j.
    """
    if d < 0:
        raise ValueError("d phải là số nguyên không âm.")

    if K == 1:
        return [(i,) for i in range(d)]
    elif K == 2:
        singletons = [(i,) for i in range(d)]
        pairs = list(combinations(range(d), 2))
        return singletons + pairs
    else:
        raise ValueError(f"Chỉ hỗ trợ K in {{1, 2}}, nhận được: K = {K}")

#Cái này là cái sinh ra X
def design_matrix(Z, K, intercept = True):
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError("Z phải là mảng 2 chiều (N, d).")
    if K not in (1, 2):
        raise ValueError(f"Chỉ hỗ trợ K in {{1, 2}}, nhận được: K = {K}")

    N, d = Z.shape  
    cols = []

    if intercept:
        cols.append(np.ones((N, 1), dtype=float))

    chi = 2.0 * Z - 1.0
    cols.append(chi)

    if K == 2:
        pairs = list(combinations(range(d), 2))
        if pairs:
            # Nhân từng cặp cột tương ứng
            pair_cols = [
                (chi[:, i] * chi[:, j])[:, np.newaxis] for i, j in pairs
            ]
            cols.append(np.hstack(pair_cols))

    return np.hstack(cols)

#sinh ma trận mặt nạ
def sample_masks(N, d, rng, p_keep=0.5):
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng phải là một instance của np.random.Generator (tạo qua np.random.default_rng)."
        )

    if N <= 0 or d <= 0:
        raise ValueError("N và d phải là các số nguyên dương.")

    return rng.binomial(n=1, p=p_keep, size=(N, d)).astype(float) # Đây là kiểu ma trận có size như này, mỗi vị trí là số lần tung ra mặt xấp, ở đây thì chỉ có n = 1 lần thử

#Hàm này tính ra beta_hat
def ols_fit(Z, y, K, cond_max= 1e8):
    Z = np.asanyarray(Z, dtype= float)
    y = np.asanyarray(y, dtype= float)

    if Z.ndim != 2:
        raise ValueError("Z phải có shape (N, d).")

    if y.ndim != 1:
        raise ValueError("y phải là vector 1 chiều.")

    N, d = Z.shape

    if y.shape[0] != N:
        raise ValueError("Z và y phải có cùng số dòng.")

    if N <= p_K(d, K):
        raise np.linalg.LinAlgError(
            f"N={N} phải lớn hơn p_K={p_K(d, K)}."
        )

    X = design_matrix(Z, K, intercept=True)

    G = (X.T @ X) / N

    cond = np.linalg.cond(G)

    if not np.isfinite(cond) or cond > cond_max:
        raise np.linalg.LinAlgError(
            "Gram matrix kém điều kiện; run chưa well-posed."
        )

    rhs = (X.T @ y) / N

    beta_full = np.linalg.solve(G, rhs)

    Ginv = np.linalg.inv(G)

    intercept = float(beta_full[0])
    beta = beta_full[1:]
    diag_ginv = np.diag(Ginv)[1:]

    return beta, intercept, diag_ginv

#Hàm này phục vụ tính Cest
def op_inf_norm(M):
    """
    ||M||_inf = max_i sum_j |M[i, j]|

    """
    M = np.asanyarray(M, dtype=float)

    if M.ndim != 2:
        raise ValueError("M phải là ma trận hai chiều")

    return float(np.max(np.sum(np.abs(M), axis= 1)))

def realized_cest(Z, K, cond_max=1e8):
    """
    Tính C_est(Z) từ Gram matrix của Walsh design thực tế.

    C_est(Z) = max(
        lambda_min(G)^(-1/2),
        ||G^(-1)||_inf
    )
    """
    Z = np.asanyarray(Z, dtype=float)

    if Z.ndim != 2:
        raise ValueError("Z phải có shape (N, d).")

    X = design_matrix(Z, K)

    N, d = Z.shape
    if N <= p_K(d, K):
        raise np.linalg.LinAlgError(
            f"N={N} phải lớn hơn p_K={p_K(d, K)}."
        )

    Gram = (X.T @ X) / N

    cond = np.linalg.cond(Gram)
    if not np.isfinite(cond) or cond > cond_max:
        raise np.linalg.LinAlgError(
                "Gram matrix kém điều kiện; C_est không xác định."
        )
    
    lambdaa = float(np.linalg.eigvalsh(Gram)[0])

    Gram_inv = np.linalg.inv(Gram)

    gamma_term = lambdaa** (-0.5)

    C_inv = op_inf_norm(Gram_inv)

    return max(gamma_term, C_inv)


"""
DETECTION FLOOR
"""
def log_pk_over_delta(d, K, delta=None, split=3):
    p_k = p_K(d, K)
    if delta is None:
        delta = 1 / p_k
    L = math.log(2*split*p_k / delta)
    return L
#TODO: thêm validation
def certified_floor(c_est, sigma_obs, m_ub, B_pop_ub, d, N, K, delta=None, split=3):
    L = log_pk_over_delta(d, K, delta=delta, split=split)

    noise_term = sigma_obs*math.sqrt(2.0 * L/N)
    leakage_subgaussian = math.sqrt(2.0 * m_ub * L/N)
    leakage_subexpotential = (2.0/3.0) * B_pop_ub * L/N

    return c_est * (
        noise_term + leakage_subgaussian + leakage_subexpotential
    )

def certified_set(beta, floor):
    beta = np.asanyarray(beta, dtype= float)

    certified = np.abs(beta) > floor

    return certified

