"""反解 Nina v4 融合：找出 jazivxt / h-blend / rounded-ref 三个组件的身份。

公式（does-breaking-ties-help notebook 描述）：
    first      = 0.50 × jazivxt + 0.50 × h-blend
    submission = 0.47 × first  + 0.54 × rounded-ref
即  final = 0.235×jazivxt + 0.235×h-blend + 0.54×rounded-ref

对每个有序三元组 (j, h, r)，分别用
  (a) 固定权重 0.235/0.235/0.54
  (b) 最小二乘自由拟合  final ≈ a·j + b·h + c·r + d
计算 max|误差|，正确组合应给出 ~1e-15 的机器精度残差。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

FINAL = "lexsort_trial/nina_v4_submission.csv"


def load_all() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    final = pd.read_csv(FINAL).set_index("id")["Will_Buy_EV"]
    final_v = final.values.astype(float)
    idx = final.index.values

    cand: dict[str, np.ndarray] = {}
    for f in sorted(glob.glob("nina_09/*") + glob.glob("nina_11/*")):
        df = pd.read_csv(f)
        key = os.path.basename(f)
        if "id" in df.columns:
            df = df.set_index("id")
            # 对齐到 final 的 id 顺序
            s = df["Will_Buy_EV"].reindex(idx)
        else:
            s = df.iloc[:, -1]
            s = pd.Series(s.values, index=idx)
        cand[key] = s.values.astype(float)
    return final_v, cand


def main() -> None:
    final, cand = load_all()
    names = list(cand.keys())
    n = len(names)
    print(f"final: {final.shape}, 候选组件 {n} 个\n")

    # 预取数组
    arrs = [cand[k] for k in names]

    # 固定权重 0.235 / 0.235 / 0.54
    fixed_results = []
    for ri in range(n):
        r = arrs[ri]
        rterm = 0.54 * r
        for ji in range(n):
            jterm = 0.235 * arrs[ji]
            for hi in range(n):
                if ji == hi:
                    continue
                blend = jterm + 0.235 * arrs[hi] + rterm
                err = float(np.max(np.abs(blend - final)))
                fixed_results.append((err, ji, hi, ri))

    fixed_results.sort(key=lambda x: x[0])
    print("=== 固定权重 0.235/0.235/0.54：最优 5 组 ===")
    for err, ji, hi, ri in fixed_results[:5]:
        print(f"  err={err:.3e}  jazivxt={names[ji][:40]}  h-blend={names[hi][:40]}  rounded-ref={names[ri][:40]}")

    # 最小二乘自由拟合（含截距）——正确三元组应 ~1e-15
    best_ls = None
    # 对每个 ri 先做 ri 与 (ji,hi) 的组合，自由拟合三个权重
    from itertools import combinations
    for ri in range(n):
        for ji, hi in combinations(range(n), 2):
            X = np.column_stack([arrs[ji], arrs[hi], arrs[ri], np.ones_like(final)])
            coef, *_ = np.linalg.lstsq(X, final, rcond=None)
            pred = X @ coef
            err = float(np.max(np.abs(pred - final)))
            if best_ls is None or err < best_ls[0]:
                best_ls = (err, ji, hi, ri, coef)

    err, ji, hi, ri, coef = best_ls
    print("\n=== 最小二乘自由拟合：最优三元组 ===")
    print(f"  err={err:.3e}")
    print(f"  jazivxt      = {names[ji]}")
    print(f"  h-blend      = {names[hi]}")
    print(f"  rounded-ref  = {names[ri]}")
    print(f"  拟合权重 a/jazivxt={coef[0]:.6f}  b/h-blend={coef[1]:.6f}  c/rounded-ref={coef[2]:.6f}  intercept={coef[3]:.2e}")


if __name__ == "__main__":
    main()
