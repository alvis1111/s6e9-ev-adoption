"""
EV Purchases — Playground Series S6E9
MLP 单独 OOF 探路 (sklearn MLPClassifier, 零安装)
================================================
在 TE+KNN 特征上训练 MLP, 看它单独能到多少 AUC:
  >= 0.94 -> 有潜力加入集成
  <  0.93 -> 大概率负贡献, 止损

运行: python -u ev_mlp.py
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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


def to_numeric(df):
    out = df.copy()
    for c in CAT_COLS:
        out[c] = out[c].cat.codes.astype(float)
    return out


def main() -> None:
    t0 = time.time()
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET].astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    log(f"[data] X {X.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))

    for fi, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = add_te(X.iloc[tr], X.iloc[va], y.iloc[tr])
        X_tr, X_va = add_knn(X_tr, X_va, y.iloc[tr])
        X_tr = to_numeric(X_tr)
        X_va = to_numeric(X_va)
        sc = StandardScaler()
        Xs_tr = sc.fit_transform(X_tr)
        Xs_va = sc.transform(X_va)
        t = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                            alpha=1e-4, batch_size=1024, max_iter=100,
                            early_stopping=True, n_iter_no_change=10,
                            validation_fraction=0.1, random_state=SEED, verbose=False)
        mlp.fit(Xs_tr, y.iloc[tr])
        oof[va] = mlp.predict_proba(Xs_va)[:, 1]
        log(f"  fold {fi}: AUC={roc_auc_score(y.iloc[va], oof[va]):.5f}  [{time.time()-t:.0f}s]")
        del X_tr, X_va, Xs_tr, Xs_va, sc, mlp; gc.collect()

    log(f"\n[MLP] OOF AUC = {roc_auc_score(y, oof):.5f}  (GBDT 参照 0.9448)  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
