"""
EV Purchases — Playground Series S6E9
伪标签实验 (Pseudo-labeling, 仅 LGB 单 seed 快速验证)
================================================
思路: 全量 train 训练 -> 预测 test 得伪标签 -> 把 test 样本(带伪标签)
     追加进训练集 -> 5-Fold CV (只在原始 train 上算 OOF) 看是否提升。

对比:
  A. 无伪标签 baseline
  B. 硬标签 weight=1.0
  C. 硬标签 weight=0.5
  D. 硬标签 weight=0.25
  E. 软标签 (regression:binary 用概率)

运行:  python ev_pseudo.py
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
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]

BASE = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=100,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbose=-1,
)


def cv_auc(X, y, w=None, n_rounds=3000, n_valid=None):
    """
    分层 5-Fold CV。n_valid: 只在 y[:n_valid] (原始 train) 上算 AUC，
    伪标签样本 (index >= n_valid) 只进训练折、绝不进验证折，避免泄漏。
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    n_valid = len(X) if n_valid is None else n_valid
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        w_tr = w[tr_idx] if w is not None else None
        d_tr = lgb.Dataset(X_tr, y_tr, weight=w_tr, categorical_feature=CAT_COLS)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        m = lgb.train(BASE, d_tr, num_boost_round=n_rounds,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_idx] = m.predict(X_va, num_iteration=m.best_iteration)
    # 只在原始 train 部分评估
    return roc_auc_score(y.iloc[:n_valid], oof[:n_valid])


def main() -> None:
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET].astype(int)
    X_test = build_features(test)

    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    # ---- 第一步: 全量训练，预测 test 得伪标签 ----
    d_all = lgb.Dataset(X, y, categorical_feature=CAT_COLS)
    m_full = lgb.train(BASE, d_all, num_boost_round=1500)  # 无早停，用固定轮数
    p_test = m_full.predict(X_test)
    hard = (p_test > 0.5).astype(int)
    print(f"[pseudo] test pred range [{p_test.min():.4f}, {p_test.max():.4f}]")
    print(f"[pseudo] hard positive rate = {hard.mean():.4f} (train {y.mean():.4f})")

    # ---- 无伪标签 baseline ----
    auc0 = cv_auc(X, y)
    print(f"\nA. baseline (no pseudo)            = {auc0:.5f}")

    # ---- 构造增强数据集 ----
    X_aug = pd.concat([X, X_test], ignore_index=True)
    n_tr = len(X)

    for label, w_val, hard_flag in [
        ("B. hard w=1.0", 1.0, True),
        ("C. hard w=0.5", 0.5, True),
        ("D. hard w=0.25", 0.25, True),
    ]:
        y_pseudo = hard if hard_flag else p_test
        y_aug = pd.concat([y, pd.Series(y_pseudo)], ignore_index=True).astype(int)
        w_aug = np.ones(len(X_aug))
        w_aug[n_tr:] = w_val
        # OOF 只算原始 train 部分 (n_valid=n_tr)，杜绝泄漏
        auc = cv_auc(X_aug, y_aug, w=w_aug, n_valid=n_tr)
        print(f"{label}          = {auc:.5f}")

    # ---- 硬标签但只用高置信样本 ----
    conf = np.maximum(p_test, 1 - p_test)  # 置信度
    thr = np.quantile(conf, 0.5)  # 取置信度前 50%
    keep = conf >= thr
    X_hc = pd.concat([X, X_test[keep]], ignore_index=True)
    y_hc = pd.concat([y, pd.Series(hard[keep])], ignore_index=True).astype(int)
    w_hc = np.ones(len(X_hc)); w_hc[n_tr:] = 0.5
    auc_hc = cv_auc(X_hc, y_hc, w=w_hc, n_valid=n_tr)
    print(f"E. hard w=0.5 (top-50% confident) = {auc_hc:.5f}")


if __name__ == "__main__":
    main()
