"""Bài thực hành: nối my_core.py với một black box xác định, có exact beta.

Chạy từ thư mục dự án bằng .venv/bin/python:
  blackbox_walkthrough.py demo --K 1 --out-dir tmp/demo_k1
  blackbox_walkthrough.py nlp --sentence 'This movie is really good.'
  blackbox_walkthrough.py cache --cache tmp/blackbox_walkthrough/cube.npz --K 2

demo chỉ kiểm tra pipeline bằng NumPy, KHÔNG phải kết quả DistilBERT.
nlp dùng wrapper của repo mẫu; mọi OLS/C_est/floor dùng my_core của bạn.
Liệt kê full cube là chi phí lấy ground truth riêng, không phải ngân sách N.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

import my_core as core


def full_cube(d):
    """Bit thứ i của số thứ tự là mặt nạ của token i."""
    return ((np.arange(1 << d)[:, None] >> np.arange(d)) & 1).astype(float)


def collect_cube(args):
    if args.mode == "cache":
        if args.cache is None:
            raise ValueError("mode cache cần --cache PATH/cube.npz")
        with np.load(args.cache, allow_pickle=False) as data:
            Z = data["Z"]
            y = data["y"]
            meta = json.loads(str(data["metadata"].item()))
        return Z, y, meta

    if args.mode == "demo":
        Z = full_cube(6)
        chi = 2 * Z - 1
        y = (0.5 + 0.14 * chi[:, 0] - 0.09 * chi[:, 1]
             + 0.07 * chi[:, 0] * chi[:, 2]
             + 0.025 * chi[:, 1] * chi[:, 2] * chi[:, 3])
        return Z, y, {
            "source": "synthetic probability demo; not an NLP model",
            "units": [f"unit_{i}" for i in range(6)],
            "sigma_obs": 0.0,
        }

    # Chỉ import/tải mô hình khi thực sự chọn mode nlp.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "BudgetLIME-main"))
    from bl_models import TextClassifier

    clf = TextClassifier(model="distilbert", dataset="sst2",
                         device=args.device, chunk_size=args.chunk_size)
    try:
        ctx = clf.encode(args.sentence)
        units = ctx["free_tokens"]
        d = len(units)
        if not 3 <= d <= args.max_d:
            raise ValueError(
                f"Câu có d={d} subword; hãy chọn câu có 3..{args.max_d} subword. "
                "Không cắt token vì sẽ thay đổi bài toán."
            )
        target = clf.target_class(ctx)  # Cố định class từ câu nguyên vẹn.
        baseline = clf.make_baseline(ctx, args.reference)
        Z = full_cube(d)
        print(f"device={clf.device}; d={d}; full cube={len(Z)} masks", flush=True)
        print(f"units={units}; fixed target={target}", flush=True)
        y = clf.query(ctx, baseline, Z, target)
        # Một kiểm tra lặp nhỏ chỉ kiểm tra tính xác định trong lần chạy này.
        y_repeat = clf.query(ctx, baseline, Z[:4], target)
        repeat_diff = float(np.max(np.abs(y_repeat - y[:4])))
        if repeat_diff > 1e-6:
            raise ValueError(
                f"Repeated-query difference={repeat_diff:.3g}; "
                "cần kiểm tra tính xác định trước khi coi cube là ground truth."
            )
        import torch
        import transformers
        meta = {
            "source": "DistilBERT deterministic probability output",
            "sentence": args.sentence,
            "model": clf.model_name,
            "model_revision": getattr(clf.model.config, "_commit_hash", None),
            "reference": args.reference,
            "target": target,
            "target_label": clf.model.config.id2label.get(target, str(target)),
            "units": units,
            "sigma_obs": 0.0,
            "repeat_max_diff": repeat_diff,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": clf.device,
        }
        return Z, y, meta
    finally:
        clf.close()


def write_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def experiment(Zc, yc, meta, budgets, K, seed, delta):
    """Exact Walsh projection -> i.i.d. masks -> OLS -> certified floor."""
    d = Zc.shape[1]
    if not 3 <= d <= 13 or not np.array_equal(Zc, full_cube(d)):
        raise ValueError("Cache phải chứa full cube đúng thứ tự, 3 <= d <= 13.")
    if yc.shape != (len(Zc),) or not np.isfinite(yc).all():
        raise ValueError("Vector xác suất cache không hợp lệ.")
    if np.any((yc < 0) | (yc > 1)) or meta.get("sigma_obs") != 0.0:
        raise ValueError("Bài thực hành này cần output xác suất xác định.")
    if len(meta["units"]) != d:
        raise ValueError("Số tên token trong cache không khớp với mặt nạ.")

    # Trên full cube, X^T X / 2^d = I: beta_S = E[g(z) chi_S(z)].
    Xc = core.design_matrix(Zc, K, intercept=True)
    beta_full_exact = Xc.T @ yc / len(Zc)
    beta_exact = beta_full_exact[1:]
    r_exact = yc - Xc @ beta_full_exact
    m_exact = float(np.mean(r_exact ** 2))
    B_exact = float(np.max(np.abs(r_exact)))
    pk = core.p_K(d, K)
    delta = 1 / pk if delta is None else delta

    # Các dòng độc lập Bernoulli(1/2), CÓ hoàn lại; không lấy prefix full cube.
    rng = np.random.default_rng(seed)
    Zbank = core.sample_masks(max(budgets), d, rng)
    cube_indices = Zbank.astype(np.int64) @ (1 << np.arange(d))
    ybank = yc[cube_indices]  # Cache tránh query mô hình lại.
    labels = [" * ".join(f"{i}:{meta['units'][i]}" for i in S)
              for S in core.feature_subsets(d, K)]
    print(f"K={K}; pK={pk}; m_exact={m_exact:.8g}; B_exact={B_exact:.8g}")
    print(f"delta={delta:.6g} PER BUDGET; sigma_obs=0; exact m/B; split=3")
    print("    N    C_est     floor  max_error  cert  false  margin_found/total")

    summaries, coefficients = [], []
    for N in budgets:
        row = dict(N=N, status="unresolved_design", C_est=None, floor=None,
                   error_inf_full=None, bound_holds=None, certified=None,
                   false_signs=None, margin_total=None, margin_recovered=None)
        try:
            beta, b0, _ = core.ols_fit(Zbank[:N], ybank[:N], K)
            cest = core.realized_cest(Zbank[:N], K)
        except np.linalg.LinAlgError as exc:
            print(f"{N:5d}  unresolved design: {exc}")
            summaries.append(row)
            continue
        floor = core.certified_floor(cest, 0.0, m_exact, B_exact,
                                     d, N, K, delta=delta, split=3)
        cert = core.certified_set(beta, floor)  # my_core trả boolean mask.
        truth_nonzero = np.abs(beta_exact) > 1e-9
        correct = truth_nonzero & (np.sign(beta) == np.sign(beta_exact))
        margin = np.abs(beta_exact) > 2 * floor
        error = float(np.max(np.abs(np.r_[b0, beta] - beta_full_exact)))
        row.update(status="ok", C_est=float(cest), floor=float(floor),
                   error_inf_full=error, bound_holds=bool(error <= floor + 1e-12),
                   certified=int(cert.sum()), false_signs=int((cert & ~correct).sum()),
                   margin_total=int(margin.sum()),
                   margin_recovered=int((margin & cert & correct).sum()))
        summaries.append(row)
        print(f"{N:5d}  {cest:7.3f}  {floor:8.5f}  {error:9.5f}"
              f"  {row['certified']:4d}  {row['false_signs']:5d}"
              f"  {row['margin_recovered']:5d}/{row['margin_total']}")
        for j, label in enumerate(labels):
            coefficients.append(dict(
                N=N, feature=label, beta_exact=float(beta_exact[j]),
                beta_hat=float(beta[j]), floor=float(floor),
                certified=bool(cert[j]), correct_sign=bool(correct[j]),
                guarantee_margin=bool(margin[j]),
            ))
    details = dict(d=d, K=K, pK=pk, seed=seed, budgets=budgets,
                   delta_per_budget=delta, split=3, m_exact=m_exact, B_exact=B_exact,
                   cube_size=len(Zc),
                   numpy_version=np.__version__, python_version=sys.version)
    return summaries, coefficients, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["demo", "nlp", "cache"])
    parser.add_argument("--sentence", default="This movie is really good.")
    parser.add_argument("--reference", choices=["mask", "pad", "zero"], default="mask")
    parser.add_argument("--K", type=int, choices=[1, 2], default=1)
    parser.add_argument("--N-ladder", default="128,512,2000,4000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta", type=float, default=None,
                        help="failure probability per budget; default 1/pK")
    parser.add_argument("--max-d", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/blackbox_walkthrough"))
    args = parser.parse_args()
    try:
        budgets = sorted(set(int(n) for n in args.N_ladder.split(",")))
        if not budgets or min(budgets) <= 0:
            raise ValueError("N-ladder cần các số nguyên dương.")
        if not 3 <= args.max_d <= 13 or args.chunk_size <= 0:
            raise ValueError("Cần 3 <= max-d <= 13 và chunk-size > 0.")
        if args.delta is not None and not 0 < args.delta < 1:
            raise ValueError("delta phải thuộc (0,1).")
        Z, y, meta = collect_cube(args)
        summaries, coefficients, details = experiment(
            Z, y, meta, budgets, args.K, args.seed, args.delta)
    except (ValueError, ModuleNotFoundError) as exc:
        parser.exit(1, f"Không chạy được: {exc}\nXem HUONG_DAN_TAI_LAP.md.\n")
    details["fresh_model_masks"] = len(Z) + 4 if args.mode == "nlp" else 0
    details["target_selection_calls"] = 1 if args.mode == "nlp" else 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "cube.npz", Z=Z, y=y,
                        metadata=np.array(json.dumps(meta, ensure_ascii=False)))
    write_csv(args.out_dir / "summary.csv", summaries)
    write_csv(args.out_dir / "coefficients.csv", coefficients)
    (args.out_dir / "metadata.json").write_text(
        json.dumps({"probe": meta, "experiment": details}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"Saved: {args.out_dir.resolve()}")
    print("N là ngân sách mô phỏng từ cache; full cube là chi phí ground truth riêng.")


if __name__ == "__main__":
    main()
