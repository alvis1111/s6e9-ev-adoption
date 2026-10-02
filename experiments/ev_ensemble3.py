"""
EV Purchases — Playground Series S6E9
三模型集成验证 (LGB + XGB + CatBoost, 单 seed)
================================================
同一套分层 5-Fold CV 下计算三模型的 OOF：
  1. 各模型单独 AUC
  2. 两两相关性 (判断集成多样性空间)
  3. 三模型最优加权搜索
并把 OOF / test 预测存盘，供后续多种子 bagging + 伪标签复用。

运行:  python ev_ensemble3.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from ev_clean import load_clean_data, TARGET
from ev_features import build_features, LGB_CATEGORY_COLS

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5

CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]

LGB_PARAMS = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=100,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbose=-1,
)
XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="auc",
    learning_rate=0.05, max_depth=6, min_child_weight=50,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    tree_method="hist", seed=SEED, n_jobs=-1,
)
CB_PARAMS = dict(
    loss_function="Logloss", eval_metric="AUC",
    learning_rate=0.05, depth=6, l2_leaf_reg=1.0,
    random_seed=SEED, verbose=False, allow_writing_files=False,
    thread_count=-1,
)


def get_folds(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(X, y))


def run_lgb(X, y, folds):
    oof, tp = np.zeros(len(X)), np.zeros(X.shape[0])
    return _run_lgb(X, y, folds)


def _run_lgb(X, y, folds):
    oof = np.zeros(len(X))
    for tr_idx, va_idx in folds:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        m = lgb.train(LGB_PARAMS, d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_idx] = m.predict(X_va, num_iteration=m.best_iteration)
    return oof


def _run_xgb(X, y, folds):
    oof = np.zeros(len(X))
    for tr_idx, va_idx in folds:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        d_tr = xgb.DMatrix(X_tr, y_tr)
        d_va = xgb.DMatrix(X_va, y_va)
        m = xgb.train(XGB_PARAMS, d_tr, num_boost_round=3000,
                      evals=[(d_va, "va")], early_stopping_rounds=100, verbose_eval=False)
        oof[va_idx] = m.predict(d_va, iteration_range=(0, m.best_iteration + 1))
    return oof


def _run_cb(X, y, folds):
    oof = np.zeros(len(X))
    for tr_idx, va_idx in folds:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        p_tr = Pool(X_tr, y_tr, cat_features=CAT_COLS)
        p_va = Pool(X_va, y_va, cat_features=CAT_COLS)
        m = CatBoostClassifier(**CB_PARAMS, iterations=3000, early_stopping_rounds=100)
        m.fit(p_tr, eval_set=p_va, use_best_model=True)
        oof[va_idx] = m.predict_proba(p_va)[:, 1]
    return oof


def main() -> None:
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET]
    X_test = build_features(test)

    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    print(f"[data] X {X.shape} | X_test {X_test.shape}\n")

    folds = get_folds(X, y)

    print("[LGB] ...")
    oof_lgb = _run_lgb(X, y, folds)
    print(f"  LGB AUC = {roc_auc_score(y, oof_lgb):.5f}")

    print("[XGB] ...")
    oof_xgb = _run_xgb(X, y, folds)
    print(f"  XGB AUC = {roc_auc_score(y, oof_xgb):.5f}")

    print("[CatBoost] ...")
    oof_cb = _run_cb(X, y, folds)
    print(f"  CB  AUC = {roc_auc_score(y, oof_cb):.5f}")

    # 相关性
    print("\n[correlation]")
    names = ["lgb", "xgb", "cb"]
    oofs = {"lgb": oof_lgb, "xgb": oof_xgb, "cb": oof_cb}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = np.corrcoef(oofs[a], oofs[b])[0, 1]
            print(f"  corr({a},{b}) = {c:.4f}")

    # 三模型权重搜索 (网格)
    print("\n[3-way blend weight search]")
    best = (0.0, None)
    for wl in np.arange(0.0, 1.001, 0.1):
        for wx in np.arange(0.0, 1.001 - wl, 0.1):
            wc = 1.0 - wl - wx
            if wc < -1e-9:
                continue
            blend = wl * oof_lgb + wx * oof_xgb + wc * oof_cb
            auc = roc_auc_score(y, blend)
            if auc > best[0]:
                best = (auc, (round(wl, 2), round(wx, 2), round(wc, 2)))
    print(f"  best AUC = {best[0]:.5f}  w=(lgb={best[1][0]}, xgb={best[1][1]}, cb={best[1][2]})")
    print(f"  equal avg = {roc_auc_score(y, (oof_lgb + oof_xgb + oof_cb) / 3):.5f}")

    # 存盘 OOF (供后续复用)
    np.save(DATA_DIR / "oof_lgb.npy", oof_lgb)
    np.save(DATA_DIR / "oof_xgb.npy", oof_xgb)
    np.save(DATA_DIR / "oof_cb.npy", oof_cb)
    y.to_numpy().tofile(DATA_DIR / "y.npy")
    print("\n[save] oof_*.npy written.")


if __name__ == "__main__":
    main()
