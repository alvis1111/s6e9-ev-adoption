"""
EV Purchases — Playground Series S6E9
参数重新调优 (在 TE+KNN 特征下)
================================================
加了 TE+KNN 后 best_iter 从 ~700 掉到 ~150, 特征变强, 最优 lr/leaves 可能已变。
预计算 5 折 TE+KNN 特征(只算一次), 然后快速扫 LGB 参数。
运行: python -u ev_param_tune.py
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
from sklearn.neighbors import NearestNeighbors

from ev_clean import load_clean_data, TARGET
from ev_features import build_features, LGB_CATEGORY_COLS

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]
TE_NUMERIC = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
SMOOTHING = 7.0
KNN_K = 150
KNN_COL = "Annual_Income_USD"

BASE = dict(objective="binary", metric="auc", learning_rate=0.05,
            num_leaves=31, max_depth=-1, min_child_samples=100,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=SEED, n_jobs=8, verbose=-1)


def log(msg: str) -> None:
    print(msg, flush=True)


def add_te(tr_df, va_df, y_tr):
    tr = tr_df.copy(); va = va_df.copy()
    gm = y_tr.mean()
    for c in TE_NUMERIC:
        agg = y_tr.groupby(tr[c]).agg(["mean", "count"])
        enc = (agg["mean"] * agg["count"] + gm * SMOOTHING) / (agg["count"] + SMOOTHING)
        tr[f"TE_{c}"] = tr[c].map(enc).fillna(gm).astype(float)
        va[f"TE_{c}"] = va[c].map(enc).fillna(gm).astype(float)
    return tr, va


def add_knn(tr_df, va_df, y_tr):
    tr = tr_df.copy(); va = va_df.copy()
    X_tr = tr[KNN_COL].to_numpy().reshape(-1, 1).astype(float)
    X_va = va[KNN_COL].to_numpy().reshape(-1, 1).astype(float)
    y_np = y_tr.to_numpy().astype(float)
    nn = NearestNeighbors(n_neighbors=KNN_K + 1, n_jobs=8)
    nn.fit(X_tr)
    _, idx = nn.kneighbors(X_tr)
    tr["KNN_enc"] = y_np[idx[:, 1:]].mean(axis=1)
    nn2 = NearestNeighbors(n_neighbors=KNN_K, n_jobs=8)
    nn2.fit(X_tr)
    _, idx_va = nn2.kneighbors(X_va)
    va["KNN_enc"] = y_np[idx_va].mean(axis=1)
    return tr, va


def precompute(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    trs, vas = [], []
    for tr, va in skf.split(X, y):
        X_tr, X_va = add_te(X.iloc[tr], X.iloc[va], y.iloc[tr])
        X_tr, X_va = add_knn(X_tr, X_va, y.iloc[tr])
        trs.append((X_tr, tr)); vas.append((X_va, va))
    return trs, vas


def eval_params(trs, vas, y, override):
    p = dict(BASE); p.update(override)
    oof = np.zeros(len(y))
    for (X_tr, tr), (X_va, va) in zip(trs, vas):
        d_tr = lgb.Dataset(X_tr, y.iloc[tr], categorical_feature=CAT_COLS)
        d_va = lgb.Dataset(X_va, y.iloc[va], reference=d_tr)
        m = lgb.train(p, d_tr, num_boost_round=3000,
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
    log(f"[data] X {X.shape}")

    trs, vas = precompute(X, y)
    log(f"[precompute] done [{time.time()-t0:.0f}s]\n")

    log(f"[cur]    lr0.05 l31        = {eval_params(trs, vas, y, {}):.5f}")

    for lr in (0.02, 0.03, 0.08):
        log(f"[lr]     lr{lr} l31        = {eval_params(trs, vas, y, {'learning_rate': lr}):.5f}")

    for nl in (15, 63, 127):
        log(f"[leaves] lr0.05 l{nl:<3}     = {eval_params(trs, vas, y, {'num_leaves': nl}):.5f}")

    log(f"[combo]  lr0.02 l63        = {eval_params(trs, vas, y, {'learning_rate': 0.02, 'num_leaves': 63}):.5f}")
    log(f"[combo]  lr0.03 l63        = {eval_params(trs, vas, y, {'learning_rate': 0.03, 'num_leaves': 63}):.5f}")
    log(f"[combo]  lr0.02 l127       = {eval_params(trs, vas, y, {'learning_rate': 0.02, 'num_leaves': 127}):.5f}")
    log(f"[combo]  lr0.05 l15 mcs50  = {eval_params(trs, vas, y, {'num_leaves': 15, 'min_child_samples': 50}):.5f}")
    log(f"[combo]  lr0.03 l31 mcs200 = {eval_params(trs, vas, y, {'learning_rate': 0.03, 'min_child_samples': 200}):.5f}")

    log(f"\n[total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
