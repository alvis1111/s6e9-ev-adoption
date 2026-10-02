"""
EV Purchases — Playground Series S6E9
目标编码调优实验 (smoothing + 组合列)
================================================
在 baseline TE (Income/Commute/Age, s=20) 基础上:
  1. smoothing 扫描 (只调 Income 的 s)
  2. 组合列 TE: Income_q10 x City_Type / Current_Car_Type / Gender

单 LGB 快速评估。运行: python -u ev_te_tune.py
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
BASE_TE = ["Annual_Income_USD", "Daily_Commute_km", "Age"]


def log(msg: str) -> None:
    print(msg, flush=True)


def lgb_params() -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=SEED, n_jobs=8, verbose=-1)


def te_fit(key_tr, y_tr, gm, smoothing):
    """用 train 数据计算平滑目标编码表(Series)。"""
    agg = y_tr.groupby(key_tr).agg(["mean", "count"])
    return (agg["mean"] * agg["count"] + gm * smoothing) / (agg["count"] + smoothing)


def te_apply(key, enc, gm):
    """把编码表应用到 key。"""
    return key.map(enc).fillna(gm).astype(float)


def run_cv(X, y, smoothing=20.0, combo_cols=()):
    """单 LGB 5 折。BASE_TE 用给定 smoothing; combo_cols 为 (组合列名, 列list) 列表。"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    cats = [c for c in CAT_COLS if c in X.columns]
    for tr, va in skf.split(X, y):
        X_tr = X.iloc[tr].copy()
        X_va = X.iloc[va].copy()
        y_tr = y.iloc[tr]
        gm = y_tr.mean()
        # 基础 TE
        for c in BASE_TE:
            enc = te_fit(X_tr[c], y_tr, gm, smoothing)
            X_tr[f"TE_{c}"] = te_apply(X_tr[c], enc, gm)
            X_va[f"TE_{c}"] = te_apply(X_va[c], enc, gm)
        # 组合列 TE
        for name, cols in combo_cols:
            key_tr = X_tr[cols].astype(str).agg("_".join, axis=1)
            key_va = X_va[cols].astype(str).agg("_".join, axis=1)
            enc = te_fit(key_tr, y_tr, gm, smoothing)
            X_tr[f"TE_{name}"] = te_apply(key_tr, enc, gm)
            X_va[f"TE_{name}"] = te_apply(key_va, enc, gm)
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
    # 组合列用的 Income 分桶 (qcut 10)
    X["Income_q10"] = pd.qcut(X["Annual_Income_USD"], 10, labels=False, duplicates="drop").astype(int)
    log(f"[data] X {X.shape}\n")

    log(f"[ref] s=20                  = {run_cv(X, y, smoothing=20.0):.5f}")

    # smoothing 扫描 (只影响 Income, 也影响 Commute/Age 但影响小)
    for s in (10.0, 50.0, 100.0, 200.0):
        log(f"[smooth] s={s:<5}            = {run_cv(X, y, smoothing=s):.5f}")

    # 组合列 TE (在 s=20 基础上加)
    combos = [
        ("Income_q10_City", ["Income_q10", "City_Type"]),
        ("Income_q10_Car", ["Income_q10", "Current_Car_Type"]),
        ("Income_q10_Gender", ["Income_q10", "Gender"]),
        ("Income_q10_City_Car", ["Income_q10", "City_Type", "Current_Car_Type"]),
    ]
    for name, cols in combos:
        log(f"[combo] + {name:<20} = {run_cv(X, y, smoothing=20.0, combo_cols=[(name, cols)]):.5f}")

    log(f"\n[total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
