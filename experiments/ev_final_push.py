"""
EV Purchases — Playground Series S6E9
最终冲刺: 伪标签 + 三模型集成 + 多种子 bagging 叠加验证
================================================
组合已验证的有效武器，测叠加效果 (无泄漏评估):
  - 伪标签: 全量 LGB 预测 test -> 硬标签 w=1.0 加入训练
  - 三模型: LGB + XGB + CatBoost, 同 seed
  - bagging: 3 个 seed 平均
只在原始 train 上算 OOF。

运行:  python ev_final_push.py
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
N_FOLDS = 5
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]
SEEDS = [42, 2026, 777]


def lgb_params(seed):
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, verbose=-1)


def xgb_params(seed):
    return dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.05,
                max_depth=6, min_child_weight=50, subsample=0.9,
                colsample_bytree=0.9, reg_lambda=1.0,
                tree_method="hist", seed=seed, n_jobs=-1)


def cb_params(seed):
    return dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.05,
                depth=6, l2_leaf_reg=1.0, random_seed=seed,
                verbose=False, allow_writing_files=False, thread_count=-1)


def folds_of(X, y, seed):
    return list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(X, y))


def oof_lgb(X, y, folds, seed):
    oof = np.zeros(len(X))
    for tr, va in folds:
        X_tr, X_va, y_tr, y_va = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        m = lgb.train(lgb_params(seed), d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X_va, num_iteration=m.best_iteration)
    return oof


def oof_xgb(X, y, folds, seed):
    oof = np.zeros(len(X))
    for tr, va in folds:
        X_tr, X_va, y_tr, y_va = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        d_tr = xgb.DMatrix(X_tr, y_tr); d_va = xgb.DMatrix(X_va, y_va)
        m = xgb.train(xgb_params(seed), d_tr, num_boost_round=3000,
                      evals=[(d_va, "va")], early_stopping_rounds=100, verbose_eval=False)
        oof[va] = m.predict(d_va, iteration_range=(0, m.best_iteration + 1))
    return oof


def oof_cb(X, y, folds, seed):
    oof = np.zeros(len(X))
    for tr, va in folds:
        X_tr, X_va, y_tr, y_va = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        p_tr = Pool(X_tr, y_tr, cat_features=CAT_COLS)
        p_va = Pool(X_va, y_va, cat_features=CAT_COLS)
        m = CatBoostClassifier(**cb_params(seed), iterations=3000, early_stopping_rounds=100)
        m.fit(p_tr, eval_set=p_va, use_best_model=True)
        oof[va] = m.predict_proba(p_va)[:, 1]
    return oof


def main() -> None:
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET].astype(int)
    X_test = build_features(test)

    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    n_tr = len(X)
    print(f"[data] X {X.shape} | X_test {X_test.shape}\n")

    # ---- 伪标签 (seed 42 全量 LGB) ----
    d_all = lgb.Dataset(X, y, categorical_feature=CAT_COLS)
    m_full = lgb.train(lgb_params(42), d_all, num_boost_round=1500)
    hard = (m_full.predict(X_test) > 0.5).astype(int)
    X_aug = pd.concat([X, X_test], ignore_index=True)
    y_aug = pd.concat([y, pd.Series(hard)], ignore_index=True).astype(int)

    # ---- 3 组配置对比 ----
    results = {}

    # (1) 无伪标签 + 无bagging + 三模型平均 (seed 42)
    folds = folds_of(X, y, 42)
    o_lgb = oof_lgb(X, y, folds, 42)
    o_xgb = oof_xgb(X, y, folds, 42)
    o_cb = oof_cb(X, y, folds, 42)
    blend = (o_lgb + o_xgb + o_cb) / 3
    results["no-pseudo, 1seed, 3model-avg"] = roc_auc_score(y, blend)
    print(f"[1] no-pseudo 1seed 3model-avg  = {results['no-pseudo, 1seed, 3model-avg']:.5f}")

    # (2) 伪标签 + 1seed + 三模型平均
    faug = folds_of(X_aug, y_aug, 42)
    o_lgb = oof_lgb(X_aug, y_aug, faug, 42)[:n_tr]
    o_xgb = oof_xgb(X_aug, y_aug, faug, 42)[:n_tr]
    o_cb = oof_cb(X_aug, y_aug, faug, 42)[:n_tr]
    blend = (o_lgb + o_xgb + o_cb) / 3
    results["pseudo, 1seed, 3model-avg"] = roc_auc_score(y, blend)
    print(f"[2] pseudo 1seed 3model-avg      = {results['pseudo, 1seed, 3model-avg']:.5f}")

    # (3) 伪标签 + 3seed bagging + 三模型平均 (LGB/XGB/CB 各 3 seed)
    lgb_bag = np.zeros(n_tr); xgb_bag = np.zeros(n_tr); cb_bag = np.zeros(n_tr)
    for s in SEEDS:
        faug_s = folds_of(X_aug, y_aug, s)
        lgb_bag += oof_lgb(X_aug, y_aug, faug_s, s)[:n_tr] / len(SEEDS)
        xgb_bag += oof_xgb(X_aug, y_aug, faug_s, s)[:n_tr] / len(SEEDS)
        cb_bag += oof_cb(X_aug, y_aug, faug_s, s)[:n_tr] / len(SEEDS)
        print(f"      seed {s} done")
    blend = (lgb_bag + xgb_bag + cb_bag) / 3
    results["pseudo, 3seed, 3model-avg"] = roc_auc_score(y, blend)
    print(f"[3] pseudo 3seed 3model-avg      = {results['pseudo, 3seed, 3model-avg']:.5f}")

    print("\n[summary]")
    for k, v in results.items():
        print(f"  {v:.5f}  {k}")


if __name__ == "__main__":
    main()
