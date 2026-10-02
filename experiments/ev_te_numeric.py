"""
EV Purchases — Playground Series S6E9
numeric 列目标编码实验 (Playground 重复值 trick)
================================================
数据检查发现 numeric 列其实是"准类别":
  Annual_Income_USD  13214 唯一值 (每个值 ~50 行)
  Daily_Commute_km    805 唯一值 (每个值 ~830 行)
  Age                  45 唯一值

对这些离散值做 fold 内平滑目标编码, 捕捉"同值样本的真实正例率"。
运行: python -u ev_te_numeric.py
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
SEED = 42
N_FOLDS = 5
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]


def log(msg: str) -> None:
    print(msg, flush=True)


def lgb_params() -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=SEED, n_jobs=8, verbose=-1)


def run_cv(X, y, extra_cat=()):
    cats = [c for c in CAT_COLS if c in X.columns] + [c for c in extra_cat if c in X.columns]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        d_tr = lgb.Dataset(X.iloc[tr], y.iloc[tr], categorical_feature=cats)
        d_va = lgb.Dataset(X.iloc[va], y.iloc[va], reference=d_tr)
        m = lgb.train(lgb_params(), d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        del d_tr, d_va, m; gc.collect()
    return roc_auc_score(y, oof)


def run_cv_with_numeric_te(X, y, te_cols, smoothing=20.0, round_to=None):
    """对 numeric 列(离散值)做 fold 内平滑目标编码。round_to: 可选先 round。"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    cats = [c for c in CAT_COLS if c in X.columns]
    for tr, va in skf.split(X, y):
        X_tr = X.iloc[tr].copy()
        X_va = X.iloc[va].copy()
        y_tr = y.iloc[tr]
        gm = y_tr.mean()
        for c in te_cols:
            key_tr = X_tr[c]
            key_va = X_va[c]
            if round_to is not None:
                key_tr = (X_tr[c] / round_to).round() * round_to
                key_va = (X_va[c] / round_to).round() * round_to
            agg = y_tr.groupby(key_tr).agg(["mean", "count"])
            enc = (agg["mean"] * agg["count"] + gm * smoothing) / (agg["count"] + smoothing)
            X_tr[f"TE_{c}"] = key_tr.map(enc).fillna(gm).astype(float)
            X_va[f"TE_{c}"] = key_va.map(enc).fillna(gm).astype(float)
        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cats)
        d_va = lgb.Dataset(X_va, y.iloc[va], reference=d_tr)
        m = lgb.train(lgb_params(), d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X_va, num_iteration=m.best_iteration)
        del d_tr, d_va, m; gc.collect()
    return roc_auc_score(y, oof)


def main() -> None:
    t0 = time.time()
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET].astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    log(f"[data] X {X.shape}\n")

    log(f"[base]                    = {run_cv(X, y):.5f}")

    log(f"[TE Income]               = {run_cv_with_numeric_te(X, y, ['Annual_Income_USD']):.5f}")
    log(f"[TE Commute]              = {run_cv_with_numeric_te(X, y, ['Daily_Commute_km']):.5f}")
    log(f"[TE Age]                  = {run_cv_with_numeric_te(X, y, ['Age']):.5f}")
    log(f"[TE Income+Commute]       = {run_cv_with_numeric_te(X, y, ['Annual_Income_USD', 'Daily_Commute_km']):.5f}")
    log(f"[TE Income+Commute+Age]   = {run_cv_with_numeric_te(X, y, ['Annual_Income_USD', 'Daily_Commute_km', 'Age']):.5f}")

    # round 版本 (Income round 到 100, Commute round 到 1)
    log(f"[TE Income r100]          = {run_cv_with_numeric_te(X, y, ['Annual_Income_USD'], round_to=100):.5f}")
    log(f"[TE Commute r1]           = {run_cv_with_numeric_te(X, y, ['Daily_Commute_km'], round_to=1):.5f}")

    log(f"\n[total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
