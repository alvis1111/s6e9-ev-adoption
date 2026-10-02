"""
EV Purchases — Playground Series S6E9
目标编码 smoothing 细扫 (单 LGB)
================================================
趋势: smoothing 越小 OOF 越高。细扫 s = 0.5 ~ 10 找最优。
运行: python -u ev_te_smooth.py
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


def run_cv(X, y, smoothing):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    cats = [c for c in CAT_COLS if c in X.columns]
    for tr, va in skf.split(X, y):
        X_tr = X.iloc[tr].copy()
        X_va = X.iloc[va].copy()
        y_tr = y.iloc[tr]
        gm = y_tr.mean()
        for c in BASE_TE:
            agg = y_tr.groupby(X_tr[c]).agg(["mean", "count"])
            enc = (agg["mean"] * agg["count"] + gm * smoothing) / (agg["count"] + smoothing)
            X_tr[f"TE_{c}"] = X_tr[c].map(enc).fillna(gm).astype(float)
            X_va[f"TE_{c}"] = X_va[c].map(enc).fillna(gm).astype(float)
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

    for s in (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
        log(f"[smooth] s={s:<5} = {run_cv(X, y, s):.5f}")

    log(f"\n[total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
