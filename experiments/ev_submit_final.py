"""
EV Purchases — Playground Series S6E9
最终方案: 三模型集成 + numeric 目标编码
================================================
突破点 (实验验证): Annual_Income_USD 是 13214 个"准类别"离散值,
fold 内平滑目标编码带来 +0.002 的 OOF 增益 (0.94201 -> 0.94414)。

方案:
  - 特征: build_features(24) + 3 个 numeric 目标编码 (Income/Commute/Age, fold 内防泄漏)
  - 模型: LGB + XGB + CatBoost, 5 折, 等权平均
  - 限线程防卡死, 实时输出, 每折释放内存

输出: submission_final.csv
运行: python -u ev_submit_final.py
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
SEED = 42
N_FOLDS = 5
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]
TE_NUMERIC = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
SMOOTHING = 7.0
KNN_K = 150
KNN_COL = "Annual_Income_USD"
OUT_NAME = "submission_final_v2.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def lgb_params() -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=SEED, n_jobs=8, verbose=-1)


def xgb_params() -> dict:
    return dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.05,
                max_depth=6, min_child_weight=50, subsample=0.9,
                colsample_bytree=0.9, reg_lambda=1.0,
                tree_method="hist", seed=SEED, n_jobs=4)


def cb_params() -> dict:
    return dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.05,
                depth=6, l2_leaf_reg=1.0, random_seed=SEED,
                verbose=False, allow_writing_files=False, thread_count=8)


def add_te(tr_df, va_df, te_df, y_tr, cols, smoothing=SMOOTHING):
    """fold 内平滑目标编码: 用 tr_df/y_tr 算, 应用到 va/te。返回 (tr, va, te) 副本(新增 TE 列)。"""
    tr = tr_df.copy()
    va = va_df.copy()
    te = te_df.copy()
    gm = y_tr.mean()
    for c in cols:
        agg = y_tr.groupby(tr[c]).agg(["mean", "count"])
        enc = (agg["mean"] * agg["count"] + gm * smoothing) / (agg["count"] + smoothing)
        tr[f"TE_{c}"] = tr[c].map(enc).fillna(gm).astype(float)
        va[f"TE_{c}"] = va[c].map(enc).fillna(gm).astype(float)
        te[f"TE_{c}"] = te[c].map(enc).fillna(gm).astype(float)
    return tr, va, te


def add_knn(tr_df, va_df, te_df, y_tr, col, k=KNN_K):
    """Income 一维 KNN 目标编码: 用 tr 的 y/income 找 k 近邻标签均值。返回副本(新增 KNN_enc 列)。"""
    tr = tr_df.copy()
    va = va_df.copy()
    te = te_df.copy()
    X_tr = tr[col].to_numpy().reshape(-1, 1).astype(float)
    X_va = va[col].to_numpy().reshape(-1, 1).astype(float)
    X_te = te[col].to_numpy().reshape(-1, 1).astype(float)
    y_np = y_tr.to_numpy().astype(float)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=8)
    nn.fit(X_tr)
    _, idx = nn.kneighbors(X_tr)
    tr["KNN_enc"] = y_np[idx[:, 1:]].mean(axis=1)
    nn2 = NearestNeighbors(n_neighbors=k, n_jobs=8)
    nn2.fit(X_tr)
    _, idx_va = nn2.kneighbors(X_va)
    _, idx_te = nn2.kneighbors(X_te)
    va["KNN_enc"] = y_np[idx_va].mean(axis=1)
    te["KNN_enc"] = y_np[idx_te].mean(axis=1)
    return tr, va, te


def main() -> None:
    t0 = time.time()
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET].astype(int)
    X_test = build_features(test)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    log(f"[data] X {X.shape} | X_test {X_test.shape}\n")

    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    oof = {"lgb": np.zeros(len(X)), "xgb": np.zeros(len(X)), "cb": np.zeros(len(X))}
    tp = {"lgb": np.zeros(len(X_test)), "xgb": np.zeros(len(X_test)), "cb": np.zeros(len(X_test))}

    for fi, (tr, va) in enumerate(folds, start=1):
        # fold 内目标编码 (防泄漏)
        X_tr, X_va, X_te = add_te(X.iloc[tr], X.iloc[va], X_test, y.iloc[tr], TE_NUMERIC)
        X_tr, X_va, X_te = add_knn(X_tr, X_va, X_te, y.iloc[tr], KNN_COL)
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        log(f"--- fold {fi}/{N_FOLDS}  (train {len(tr)}, val {len(va)}) ---")

        dtest_xgb = xgb.DMatrix(X_te)
        ptest_cb = Pool(X_te, cat_features=CAT_COLS)

        # LGB
        t = time.time()
        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        m = lgb.train(lgb_params(), d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        bi = m.best_iteration
        oof["lgb"][va] = m.predict(X_va, num_iteration=bi)
        tp["lgb"] += m.predict(X_te, num_iteration=bi) / N_FOLDS
        del d_tr, d_va, m; gc.collect()
        log(f"    LGB {time.time()-t:5.0f}s  best_iter={bi}")

        # XGB
        t = time.time()
        d_tr = xgb.DMatrix(X_tr, y_tr); d_va = xgb.DMatrix(X_va, y_va)
        m = xgb.train(xgb_params(), d_tr, num_boost_round=3000,
                      evals=[(d_va, "va")], early_stopping_rounds=100, verbose_eval=False)
        bi = m.best_iteration
        oof["xgb"][va] = m.predict(d_va, iteration_range=(0, bi + 1))
        tp["xgb"] += m.predict(dtest_xgb, iteration_range=(0, bi + 1)) / N_FOLDS
        del d_tr, d_va, m; gc.collect()
        log(f"    XGB {time.time()-t:5.0f}s  best_iter={bi}")

        # CatBoost
        t = time.time()
        p_tr = Pool(X_tr, y_tr, cat_features=CAT_COLS); p_va = Pool(X_va, y_va, cat_features=CAT_COLS)
        m = CatBoostClassifier(**cb_params(), iterations=3000, early_stopping_rounds=100)
        m.fit(p_tr, eval_set=p_va, use_best_model=True)
        bi = m.get_best_iteration()
        oof["cb"][va] = m.predict_proba(p_va)[:, 1]
        tp["cb"] += m.predict_proba(ptest_cb)[:, 1] / N_FOLDS
        del p_tr, p_va, m; gc.collect()
        log(f"    CB  {time.time()-t:5.0f}s  best_iter={bi}")

        fb = (oof["lgb"][va] + oof["xgb"][va] + oof["cb"][va]) / 3
        log(f"    fold AUC  lgb={roc_auc_score(y.iloc[va], oof['lgb'][va]):.5f} "
            f"xgb={roc_auc_score(y.iloc[va], oof['xgb'][va]):.5f} "
            f"cb={roc_auc_score(y.iloc[va], oof['cb'][va]):.5f} "
            f"blend={roc_auc_score(y.iloc[va], fb):.5f}")

    # ---- 汇总 ----
    log("\n[OOF] ...")
    for k in oof:
        log(f"    {k:3s} = {roc_auc_score(y, oof[k]):.5f}")
    blend = (oof["lgb"] + oof["xgb"] + oof["cb"]) / 3
    log(f"    blend equal = {roc_auc_score(y, blend):.5f}")

    # ---- 生成提交 ----
    tp_blend = (tp["lgb"] + tp["xgb"] + tp["cb"]) / 3
    sub["Will_Buy_EV"] = tp_blend
    out = DATA_DIR / OUT_NAME
    sub.to_csv(out, index=False)
    log(f"\n[save] {out}  pred [{tp_blend.min():.4f}, {tp_blend.max():.4f}]  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
