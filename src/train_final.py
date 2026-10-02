"""S6E9 —— 生成最终提交：Naji 特征工程 + 多 seed LightGBM 平均。

用法：
    python train_final.py        # 用 100% 训练数据，3 seed 平均，输出提交 CSV

依赖 naji_features.py（特征工程）与 train.csv / test.csv / EV_orig.csv。
输出：submission_s6e9_final_multiseed.csv（OOF AUC ≈ 0.94626）。
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from naji_features import build_features, make_lgb, te_transform

SEEDS = [42, 2024, 2026]
TARGET = "Will_Buy_EV"
OUT = "submission_s6e9_final_multiseed.csv"


def main() -> None:
    t0 = time.time()
    X, y, X_test, ids, TE_COLS = build_features()
    print(f"特征 {X.shape[1]}，TE 列 {len(TE_COLS)}", flush=True)

    oof_all = np.zeros((len(SEEDS), len(X)))
    test_all = np.zeros((len(SEEDS), len(X_test)))

    for si, seed in enumerate(SEEDS):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        te_pred = np.zeros(len(X_test))
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx].copy(), y.iloc[va_idx]
            X_te = X_test.copy()
            X_tr, X_va, X_te = te_transform(X_tr, y_tr, X_va, X_te, TE_COLS)
            clf = make_lgb(seed)
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(500, verbose=False)])
            oof[va_idx] = clf.predict_proba(X_va)[:, 1]
            te_pred += clf.predict_proba(X_te)[:, 1] / 5
        oof_all[si] = oof
        test_all[si] = te_pred
        print(f"  seed {seed}: OOF {roc_auc_score(y, oof):.5f}  [{time.time()-t0:.0f}s]", flush=True)

    oof_mean = oof_all.mean(axis=0)
    test_mean = test_all.mean(axis=0)
    print(f"\n平均 OOF AUC: {roc_auc_score(y, oof_mean):.5f}", flush=True)

    pd.DataFrame({"id": ids, TARGET: test_mean}).to_csv(OUT, index=False)
    print(f"已保存 {OUT}（{len(test_mean)} 行）", flush=True)


if __name__ == "__main__":
    main()
