"""
EV Purchases — Playground Series S6E9
超参数调优 (Hyperparameter tuning)
================================================
在特征工程结果 (OOF 0.94187) 之上，用同一套分层 5-Fold CV
对比若干组 LightGBM 参数，找最优配置。

运行:  python ev_tune.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from ev_clean import load_clean_data, TARGET
from ev_features import build_features, LGB_CATEGORY_COLS

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5

# 候选参数组
CONFIGS = [
    dict(name="lr0.02_leaves63",   learning_rate=0.02, num_leaves=63,  min_child_samples=50),
    dict(name="lr0.02_leaves127",  learning_rate=0.02, num_leaves=127, min_child_samples=50),
    dict(name="lr0.01_leaves63",   learning_rate=0.01, num_leaves=63,  min_child_samples=50),
    dict(name="lr0.05_leaves31",   learning_rate=0.05, num_leaves=31,  min_child_samples=100),
    dict(name="lr0.02_leaves127_mc20", learning_rate=0.02, num_leaves=127, min_child_samples=20),
    dict(name="lr0.02_leaves255_mc20", learning_rate=0.02, num_leaves=255, min_child_samples=20),
]

BASE_PARAMS = dict(
    objective="binary",
    metric="auc",
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def run_cv(X, y, cat_cols, params, num_boost_round=10000, early_stop=200):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        m = lgb.train(
            params, d_tr, num_boost_round=num_boost_round,
            valid_sets=[d_va], callbacks=[lgb.early_stopping(early_stop, verbose=False)],
        )
        oof[va_idx] = m.predict(X_va, num_iteration=m.best_iteration)
    return roc_auc_score(y, oof)


def main() -> None:
    train, _, _ = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET]
    cat_cols = LGB_CATEGORY_COLS + ["Income_bin"]

    print(f"[data] X {X.shape}\n")
    results = []
    for cfg in CONFIGS:
        name = cfg.pop("name")
        p = {**BASE_PARAMS, **cfg}
        auc = run_cv(X, y, cat_cols, p)
        results.append((name, auc))
        print(f"  {name:24s} OOF AUC = {auc:.5f}")

    results.sort(key=lambda t: -t[1])
    print("\n[best]", results[0][0], f"{results[0][1]:.5f}")
    print("\n[ranking]")
    for name, auc in results:
        print(f"  {auc:.5f}  {name}")


if __name__ == "__main__":
    main()
