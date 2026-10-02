"""
EV Purchases — Playground Series S6E9
单 LGB + 伪标签 5-Fold —— 快速保底提交
================================================
与 ev_submit_ensemble.py 的 LGB 分支完全一致:
  - 伪标签: 全量 LGB seed42 (1500 轮) 硬标签 > 0.5
  - 5-Fold 在增强集上训练 LGB (n_jobs=4, 防卡死)
  - 只在原始 train 上算 OOF (无泄漏)
输出: submission_lgb.csv

用途: 三模型集成(CB 慢)跑完前的保底提交。
运行: python -u ev_submit_lgb.py
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from ev_clean import load_clean_data, TARGET
from ev_features import build_features, LGB_CATEGORY_COLS

DATA_DIR = Path(__file__).resolve().parent
N_FOLDS = 5
SEED = 42
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]
OUT_NAME = "submission_lgb.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def lgb_params(seed: int) -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=seed, n_jobs=4, verbose=-1)


def main() -> None:
    t0 = time.time()
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET].astype(int)
    X_test = build_features(test)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    n_tr = len(X)
    log(f"[data] X {X.shape} | X_test {X_test.shape}")

    # 伪标签
    t = time.time()
    d_all = lgb.Dataset(X, y, categorical_feature=CAT_COLS)
    m_full = lgb.train(lgb_params(SEED), d_all, num_boost_round=1500)
    hard = (m_full.predict(X_test) > 0.5).astype(int)
    log(f"[pseudo] hard rate {hard.mean():.4f}  [{time.time()-t:.0f}s]")
    del d_all, m_full
    gc.collect()

    X_aug = pd.concat([X, X_test], ignore_index=True)
    y_aug = pd.concat([y, pd.Series(hard)], ignore_index=True).astype(int)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(n_tr)
    tp = np.zeros(len(X_test))

    for fi, (tr, va) in enumerate(skf.split(X_aug, y_aug), start=1):
        va_orig = va[va < n_tr]
        d_tr = lgb.Dataset(X_aug.iloc[tr], y_aug.iloc[tr], categorical_feature=CAT_COLS)
        d_va = lgb.Dataset(X_aug.iloc[va], y_aug.iloc[va], reference=d_tr)
        m = lgb.train(lgb_params(SEED), d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        bi = m.best_iteration
        oof[va_orig] = m.predict(X_aug.iloc[va_orig], num_iteration=bi)
        tp += m.predict(X_test, num_iteration=bi) / N_FOLDS
        fa = roc_auc_score(y.iloc[va_orig], oof[va_orig])
        log(f"  fold {fi}: AUC={fa:.5f}  best_iter={bi}  [{time.time()-t0:.0f}s]")
        del d_tr, d_va, m
        gc.collect()

    log(f"[OOF] AUC = {roc_auc_score(y, oof):.5f}")
    sub["Will_Buy_EV"] = tp
    out = DATA_DIR / OUT_NAME
    sub.to_csv(out, index=False)
    log(f"[save] {out}  pred [{tp.min():.4f}, {tp.max():.4f}]  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
