"""公开 OOF 库「冲最高分」融合器。

思路：头部选手的「抄」不是盲凑，而是——
1. 用训练集 OOF + 真实标签，优化各模型的融合权重（最大化训练 AUC）
2. 把权重套到对应的测试预测上，产出 submission

做法：rank 变换对齐尺度 + 权重搜索（防 0.999 相关性下的过拟合，用 rank 融合 + 简单权重搜索）。

用法：
    python blend_oof.py --oof_dir <含 *_oof*.csv 的目录> --test_dir <含 *_test*.csv 的目录>
输出：submission_blend_oof.csv
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

TRAIN = "S6E9_EV_Adoption/train.csv"


def load_train_labels() -> np.ndarray:
    return pd.read_csv(TRAIN)["Will_Buy_EV"].map({"Yes": 1, "No": 0}).values.astype(int)


def read_pred(path: str, n: int) -> np.ndarray:
    """读一个预测文件，返回 n 行的 float 数组（对齐 train 或 test 顺序）。"""
    if path.endswith(".npy"):
        return np.load(path).astype(float)
    df = pd.read_csv(path)
    # 优先用 oof/pred 列；否则用最后一列
    for col in ["oof", "prediction", "Will_Buy_EV", "pred", "target"]:
        if col in df.columns:
            return df[col].values.astype(float)
    return df.iloc[:, -1].values.astype(float)


def rank_norm(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / len(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", default="submission_blend_oof.csv")
    args = ap.parse_args()

    y = load_train_labels()
    n_tr = len(y)

    oof_files = sorted(glob.glob(os.path.join(args.oof_dir, "**", "*oof*"), recursive=True))
    oof_files = [f for f in oof_files if f.endswith((".csv", ".npy"))]
    test_files = sorted(glob.glob(os.path.join(args.test_dir, "**", "*test*"), recursive=True))
    test_files = [f for f in test_files if f.endswith((".csv", ".npy"))]

    print(f"训练 OOF 文件 {len(oof_files)} 个，测试预测文件 {len(test_files)} 个\n")

    oofs, names = [], []
    for f in oof_files:
        try:
            v = read_pred(f, n_tr)
            if len(v) != n_tr:
                continue
            oofs.append(rank_norm(v))
            names.append(os.path.basename(f).replace("_oof", "").replace(".csv", "").replace(".npy", ""))
            auc = roc_auc_score(y, v)
            print(f"  [oof] {os.path.basename(f):45s} 单独 AUC={auc:.5f}")
        except Exception as e:
            print(f"  [skip] {f}: {e}")

    if len(oofs) < 2:
        print("OOF 文件不足，退出。")
        return

    O = np.column_stack(oofs)  # (n_tr, m)

    # 1) 等权 rank 平均
    eq = O.mean(axis=1)
    eq_auc = roc_auc_score(y, eq)
    print(f"\n等权 rank 平均: AUC={eq_auc:.5f}  ({len(oofs)} 模型)")

    # 2) 按各模型单独 AUC 加权
    solo = np.array([roc_auc_score(y, o) for o in oofs])
    w_solo = solo / solo.sum()
    sw = O @ w_solo
    sw_auc = roc_auc_score(y, sw)
    print(f"按单独 AUC 加权: AUC={sw_auc:.5f}")

    # 3) 贪心前向选择（每次加入最提升 AUC 的模型，权重用简单网格）
    print("\n贪心前向融合:")
    selected: list[int] = []
    best_auc = 0.0
    for _ in range(min(20, len(oofs))):
        best_gain, best_i = 0.0, None
        for i in range(len(oofs)):
            if i in selected:
                continue
            cand = selected + [i]
            sub = O[:, cand].mean(axis=1)
            a = roc_auc_score(y, sub)
            if a > best_gain:
                best_gain, best_i = a, i
        if best_i is None or best_gain <= best_auc + 1e-6:
            break
        selected.append(best_i)
        best_auc = best_gain
        print(f"  + {names[best_i]:40s} -> AUC={best_auc:.5f}")

    final = O[:, selected].mean(axis=1)
    print(f"\n贪心最优 {len(selected)} 模型: 训练 AUC={roc_auc_score(y, final):.5f}")

    # 4) 应用权重到测试
    test_loaded = []
    for f in test_files:
        try:
            v = read_pred(f, 286571)
            if len(v) != 286571:
                continue
            test_loaded.append((os.path.basename(f), rank_norm(v)))
        except Exception as e:
            print(f"  [skip test] {f}: {e}")

    print(f"\n测试预测文件 {len(test_loaded)} 个")

    # 建立 oof 名 -> test 名 的映射（近似匹配，去掉版本后缀）
    def key(name: str) -> str:
        return name.lower().replace("_", "").replace("-", "").replace(" ", "")

    test_map = {key(tn): tv for tn, tv in test_loaded}

    # 用贪心选出的模型名去匹配 test 文件
    chosen_names = [names[i] for i in selected]
    final_test = None
    matched = 0
    for cn in chosen_names:
        tv = test_map.get(key(cn))
        if tv is not None:
            final_test = tv if final_test is None else final_test + tv
            matched += 1
        else:
            # 模糊匹配：名字含共同子串
            for tn, tv in test_loaded:
                if key(cn)[:12] in key(tn) or key(tn)[:12] in key(cn):
                    final_test = tv if final_test is None else final_test + tv
                    matched += 1
                    break

    if final_test is None or matched == 0:
        print("\n!!! 无法把 OOF 模型名匹配到测试文件，改用全部测试文件的等权 rank 平均。")
        final_test = np.mean([tv for _, tv in test_loaded], axis=0)
    else:
        final_test = final_test / matched
        print(f"匹配到 {matched} 个测试文件，取 rank 平均")

    # 输出：rank 平均后映射回 [0,1]
    ids = pd.read_csv("S6E9_EV_Adoption/sample_submission.csv")["id"]
    pred = (rankdata(final_test) - 1) / (len(final_test) - 1)
    out = pd.DataFrame({"id": ids, "Will_Buy_EV": pred})
    out.to_csv(args.out, index=False)
    print(f"\n已保存 {args.out}（{len(out)} 行）")


if __name__ == "__main__":
    main()
