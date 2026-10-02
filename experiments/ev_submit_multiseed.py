"""
EV Purchases — Playground Series S6E9
多 seed bagging + 三模型加权 (最终冲刺)
================================================
在 v2 (同值TE s=7 + KNN k=150 + 三模型) 基础上:
  - 3 个 seed 各跑一遍 (每个 seed 独立 fold 内 TE/KNN, 防泄漏)
  - 多 seed 平均各模型 OOF/test 预测
  - 三模型权重搜索 (在平均 OOF 上) + 等权

输出:
  submission_multiseed.csv      (等权)
  submission_multiseed_w.csv    (加权)

运行: python -u ev_submit_multiseed.py
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from ev_clean import load_clean_data, TARGET
from ev_features import build_features, LGB_CATEGORY_COLS

DATA_DIR = Path(__file__).resolve().parent
N_FOLDS = 5
SEEDS = [42, 2026, 777]
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]
TE_NUMERIC = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
SMOOTHING = 7.0
KNN_K = 150
KNN_COL = "Annual_Income_USD"


def log(msg: str) -> None:
    print(msg, flush=True)


def lgb_params(seed) -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=seed, n_jobs=8, verbose=-1)


def xgb_params(seed) -> dict:
    return dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.05,
                max_depth=6, min_child_weight=50, subsample=0.9,
                colsample_bytree=0.9, reg_lambda=1.0,
                tree_method="hist", seed=seed, n_jobs=4)


def cb_params(seed) -> dict:
    return dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.05,
                depth=6, l2_leaf_reg=1.0, random_seed=seed,
                verbose=False, allow_writing_files=False, thread_count=8)


def add_te(tr_df, va_df, te_df, y_tr):
    tr = tr_df.copy(); va = va_df.copy(); te = te_df.copy()
    gm = y_tr.mean()
    for c in TE_NUMERIC:
        agg = y_tr.groupby(tr[c]).agg(["mean", "count"])
        enc = (agg["mean"] * agg["count"] + gm * SMOOTHING) / (agg["count"] + SMOOTHING)
        tr[f"TE_{c}"] = tr[c].map(enc).fillna(gm).astype(float)
        va[f"TE_{c}"] = va[c].map(enc).fillna(gm).astype(float)
        te[f"TE_{c}"] = te[c].map(enc).fillna(gm).astype(float)
    return tr, va, te


def add_knn(tr_df, va_df, te_df, y_tr):
    tr = tr_df.copy(); va = va_df.copy(); te = te_df.copy()
    X_tr = tr[KNN_COL].to_numpy().reshape(-1, 1).astype(float)
    X_va = va[KNN_COL].to_numpy().reshape(-1, 1).astype(float)
    X_te = te[KNN_COL].to_numpy().reshape(-1, 1).astype(float)
    y_np = y_tr.to_numpy().astype(float)
    nn = NearestNeighbors(n_neighbors=KNN_K + 1, n_jobs=8)
    nn.fit(X_tr)
    _, idx = nn.kneighbors(X_tr)
    tr["KNN_enc"] = y_np[idx[:, 1:]].mean(axis=1)
    nn2 = NearestNeighbors(n_neighbors=KNN_K, n_jobs=8)
    nn2.fit(X_tr)
    _, idx_va = nn2.kneighbors(X_va)
    _, idx_te = nn2.kneighbors(X_te)
    va["KNN_enc"] = y_np[idx_va].mean(axis=1)
    te["KNN_enc"] = y_np[idx_te].mean(axis=1)
    return tr, va, te


def run_seed(X, y, X_test, seed):
    """单 seed 的三模型 5 折, 返回 (oof_dict, tp_dict)。"""
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(X, y))
    oof = {"lgb": np.zeros(len(X)), "xgb": np.zeros(len(X)), "cb": np.zeros(len(X))}
    tp = {"lgb": np.zeros(len(X_test)), "xgb": np.zeros(len(X_test)), "cb": np.zeros(len(X_test))}

    for tr, va in folds:
        X_tr, X_va, X_te = add_te(X.iloc[tr], X.iloc[va], X_test, y.iloc[tr])
        X_tr, X_va, X_te = add_knn(X_tr, X_va, X_te, y.iloc[tr])
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        dtest_xgb = xgb.DMatrix(X_te)
        ptest_cb = Pool(X_te, cat_features=CAT_COLS)

        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        m = lgb.train(lgb_params(seed), d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof["lgb"][va] = m.predict(X_va, num_iteration=m.best_iteration)
        tp["lgb"] += m.predict(X_te, num_iteration=m.best_iteration) / N_FOLDS
        del d_tr, d_va, m; gc.collect()

        d_tr = xgb.DMatrix(X_tr, y_tr); d_va = xgb.DMatrix(X_va, y_va)
        m = xgb.train(xgb_params(seed), d_tr, num_boost_round=3000,
                      evals=[(d_va, "va")], early_stopping_rounds=100, verbose_eval=False)
        oof["xgb"][va] = m.predict(d_va, iteration_range=(0, m.best_iteration + 1))
        tp["xgb"] += m.predict(dtest_xgb, iteration_range=(0, m.best_iteration + 1)) / N_FOLDS
        del d_tr, d_va, m; gc.collect()

        p_tr = Pool(X_tr, y_tr, cat_features=CAT_COLS); p_va = Pool(X_va, y_va, cat_features=CAT_COLS)
        m = CatBoostClassifier(**cb_params(seed), iterations=3000, early_stopping_rounds=100)
        m.fit(p_tr, eval_set=p_va, use_best_model=True)
        oof["cb"][va] = m.predict_proba(p_va)[:, 1]
        tp["cb"] += m.predict_proba(ptest_cb)[:, 1] / N_FOLDS
        del p_tr, p_va, m; gc.collect()

    return oof, tp


def main() -> None:
    t0 = time.time()
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET].astype(int)
    X_test = build_features(test)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    log(f"[data] X {X.shape}\n")

    oof_acc = {k: np.zeros(len(X)) for k in ("lgb", "xgb", "cb")}
    tp_acc = {k: np.zeros(len(X_test)) for k in ("lgb", "xgb", "cb")}

    for seed in SEEDS:
        log(f"[seed {seed}] ...")
        oof, tp = run_seed(X, y, X_test, seed)
        for k in oof_acc:
            oof_acc[k] += oof[k] / len(SEEDS)
            tp_acc[k] += tp[k] / len(SEEDS)
        b = (oof["lgb"] + oof["xgb"] + oof["cb"]) / 3
        log(f"    seed blend OOF = {roc_auc_score(y, b):.5f}  [{time.time()-t0:.0f}s]")

    # ---- 汇总 ----
    log("\n[multi-seed OOF] ...")
    for k in oof_acc:
        log(f"    {k:3s} = {roc_auc_score(y, oof_acc[k]):.5f}")
    blend_eq = (oof_acc["lgb"] + oof_acc["xgb"] + oof_acc["cb"]) / 3
    log(f"    blend equal = {roc_auc_score(y, blend_eq):.5f}")

    # 权重搜索
    best = (0.0, None)
    for wl in np.arange(0.0, 1.001, 0.05):
        for wx in np.arange(0.0, 1.001 - wl, 0.05):
            wc = 1.0 - wl - wx
            if wc < -1e-9:
                continue
            b = wl * oof_acc["lgb"] + wx * oof_acc["xgb"] + wc * oof_acc["cb"]
            a = roc_auc_score(y, b)
            if a > best[0]:
                best = (a, (round(wl, 2), round(wx, 2), round(wc, 2)))
    log(f"    best weight = {best[0]:.5f}  w={best[1]}")

    # ---- 提交 ----
    tp_eq = (tp_acc["lgb"] + tp_acc["xgb"] + tp_acc["cb"]) / 3
    sub["Will_Buy_EV"] = tp_eq
    out = DATA_DIR / "submission_multiseed.csv"
    sub.to_csv(out, index=False)
    log(f"\n[save] {out}")

    wl, wx, wc = best[1]
    tp_w = wl * tp_acc["lgb"] + wx * tp_acc["xgb"] + wc * tp_acc["cb"]
    sub["Will_Buy_EV"] = tp_w
    out2 = DATA_DIR / "submission_multiseed_w.csv"
    sub.to_csv(out2, index=False)
    log(f"[save] {out2}  (w={best[1]})")
    log(f"[total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
