"""
EV Purchases — Playground Series S6E9
KNN 目标编码实验 (相似值聚合 vs 同值聚合)
================================================
同值 TE 只聚合"Income 完全相同"的样本(~50个)。KNN 聚合"Income 最近"的 K 个
样本, 捕捉 VAE 数据里"相近 Income -> 相近模板"的连续性信号。

在 同值 TE (Income/Commute/Age, s=7) 基础上, 加 KNN 编码, 看是否再涨。
运行: python -u ev_knn_te.py
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
BASE_TE = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
SMOOTHING = 7.0


def log(msg: str) -> None:
    print(msg, flush=True)


def lgb_params() -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=SEED, n_jobs=8, verbose=-1)


def knn_enc_train(y_tr, X_tr, k):
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=8)
    nn.fit(X_tr)
    _, idx = nn.kneighbors(X_tr)
    idx = idx[:, 1:]  # 去掉自己
    return y_tr[idx].mean(axis=1)


def knn_enc_query(y_tr, X_tr, X_q, k):
    nn = NearestNeighbors(n_neighbors=k, n_jobs=8)
    nn.fit(X_tr)
    _, idx = nn.kneighbors(X_q)
    return y_tr[idx].mean(axis=1)


def run_cv(X, y, knn_cols, k):
    """同值 TE (s=7) + KNN 编码(knn_cols 上), 单 LGB 5 折。"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    cats = [c for c in CAT_COLS if c in X.columns]
    for tr, va in skf.split(X, y):
        X_tr = X.iloc[tr].copy()
        X_va = X.iloc[va].copy()
        y_tr = y.iloc[tr]
        gm = y_tr.mean()
        # 同值 TE
        for c in BASE_TE:
            agg = y_tr.groupby(X_tr[c]).agg(["mean", "count"])
            enc = (agg["mean"] * agg["count"] + gm * SMOOTHING) / (agg["count"] + SMOOTHING)
            X_tr[f"TE_{c}"] = X_tr[c].map(enc).fillna(gm).astype(float)
            X_va[f"TE_{c}"] = X_va[c].map(enc).fillna(gm).astype(float)
        # KNN 编码 (knn_cols 上, 标准化后)
        Xk_tr = X_tr[knn_cols].to_numpy().astype(float)
        Xk_va = X_va[knn_cols].to_numpy().astype(float)
        # 标准化 (fold 内)
        mu = Xk_tr.mean(axis=0); sd = Xk_tr.std(axis=0) + 1e-9
        Xk_tr = (Xk_tr - mu) / sd
        Xk_va = (Xk_va - mu) / sd
        y_tr_np = y_tr.to_numpy().astype(float)
        X_tr["KNN_enc"] = knn_enc_train(y_tr_np, Xk_tr, k)
        X_va["KNN_enc"] = knn_enc_query(y_tr_np, Xk_tr, Xk_va, k)
        # 训练
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

    for k in (150, 200, 300, 500):
        log(f"[KNN Income] k={k:<4}       = {run_cv(X, y, ['Annual_Income_USD'], k):.5f}")

    log(f"\n[total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
