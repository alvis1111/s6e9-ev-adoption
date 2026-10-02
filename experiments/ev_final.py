"""
EV Purchases — Playground Series S6E9
最终版提交 (Final submission)
================================================
= 特征工程 (ev_features.build_features) + 调优后的最优参数 (ev_tune 结果)

最优配置:  learning_rate=0.05, num_leaves=31, min_child_samples=100
           (OOF AUC = 0.94201，调参网格中最高)

输出:  submission.csv  (可直接上传 Kaggle)
运行:  python ev_final.py
"""
from __future__ import annotations

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

# 调优后的最优参数
PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def main() -> None:
    train, test, sub = load_clean_data(DATA_DIR)

    X = build_features(train)
    y = train[TARGET]
    X_test = build_features(test)

    cat_cols = LGB_CATEGORY_COLS + ["Income_bin"]

    print(f"[data] X {X.shape} | X_test {X_test.shape}")
    print(f"[params] {PARAMS['learning_rate']=} {PARAMS['num_leaves']=} {PARAMS['min_child_samples']=}\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    print("[CV] stratified 5-fold ...")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        m = lgb.train(
            PARAMS, d_tr, num_boost_round=3000,
            valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        oof[va_idx] = m.predict(X_va, num_iteration=m.best_iteration)
        test_pred += m.predict(X_test, num_iteration=m.best_iteration) / N_FOLDS

        print(f"  fold {fold}: AUC = {roc_auc_score(y_va, oof[va_idx]):.5f}  (best_iter={m.best_iteration})")

    oof_auc = roc_auc_score(y, oof)
    print(f"\n[OOF] mean AUC = {oof_auc:.5f}")

    sub["Will_Buy_EV"] = test_pred
    out = DATA_DIR / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"[save] {out}  (pred range [{test_pred.min():.4f}, {test_pred.max():.4f}])")


if __name__ == "__main__":
    main()
