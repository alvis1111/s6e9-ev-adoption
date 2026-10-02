"""
EV Purchases — Playground Series S6E9
多模型集成 (Ensemble): LightGBM + XGBoost
================================================
用同一套分层 5-Fold CV，分别得到 LGB 与 XGB 的 out-of-fold 预测，
再网格搜索最优加权，输出集成 OOF AUC。

先单 seed 验证集成收益；若有效再扩展多种子 bagging。

运行:  python ev_ensemble.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
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
    tree_method="hist", enable_categorical=True,
    seed=SEED, n_jobs=-1,
)


def get_folds(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(X, y))


def run_lgb(X, y, folds):
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


def run_xgb(X, y, folds):
    oof = np.zeros(len(X))
    for tr_idx, va_idx in folds:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        d_tr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
        d_va = xgb.DMatrix(X_va, y_va, enable_categorical=True)
        m = xgb.train(XGB_PARAMS, d_tr, num_boost_round=3000,
                      evals=[(d_va, "va")], early_stopping_rounds=100, verbose_eval=False)
        oof[va_idx] = m.predict(d_va, iteration_range=(0, m.best_iteration + 1))
    return oof


def main() -> None:
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET]

    print(f"[data] X {X.shape}\n")

    # 确保 category 列是 category dtype (XGB 需要)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")

    folds = get_folds(X, y)

    print("[LGB] running ...")
    oof_lgb = run_lgb(X, y, folds)
    print(f"  LGB OOF AUC = {roc_auc_score(y, oof_lgb):.5f}")

    print("[XGB] running ...")
    oof_xgb = run_xgb(X, y, folds)
    print(f"  XGB OOF AUC = {roc_auc_score(y, oof_xgb):.5f}")

    # 权重搜索
    print("\n[blend] weight search ...")
    best = (0.0, None)
    for w in np.arange(0.0, 1.001, 0.05):
        auc = roc_auc_score(y, w * oof_lgb + (1 - w) * oof_xgb)
        if auc > best[0]:
            best = (auc, round(w, 2))
        if w in (0.0, 0.25, 0.5, 0.75, 1.0):
            print(f"  w_lgb={w:.2f}  AUC={auc:.5f}")
    print(f"\n[best blend] w_lgb={best[1]}  AUC={best[0]:.5f}")

    # 简单平均
    print(f"[equal avg] AUC = {roc_auc_score(y, (oof_lgb + oof_xgb) / 2):.5f}")

    # 相关性 (判断集成增益空间)
    corr = np.corrcoef(oof_lgb, oof_xgb)[0, 1]
    print(f"[corr lgb/xgb] {corr:.4f}")


if __name__ == "__main__":
    main()
