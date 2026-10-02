"""
EV Purchases — Playground Series S6E9
特征工程实验 (快速评估, 单 LGB)
================================================
逐组测试新特征在单 LGB 5-Fold OOF 上的增益, 找出有效特征。

配置:
  A. baseline (现有 24 特征)
  B. + 比率/密度特征
  C. + 三阶交互
  D. + 分箱特征
  E. + 目标编码 (防泄漏, fold 内平滑编码)

运行: python -u ev_feat_exp.py
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

from ev_clean import load_clean_data, TARGET, NOMINAL_CATS
from ev_features import build_features, LGB_CATEGORY_COLS

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]


def log(msg: str) -> None:
    print(msg, flush=True)


def lgb_params() -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=SEED, n_jobs=8, verbose=-1)


def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    """比率/密度特征 (无泄漏)。"""
    Z = X.copy()
    Z["Charging_Home_Ratio"] = X["Charging_Stations_Near_Home"] / (
        X["Charging_Stations_Near_Home"] + X["Charging_Stations_Near_Work"] + 1.0)
    Z["Charging_per_Car"] = X["Total_Charging_Stations"] / (X["Number_of_Cars_Owned"] + 1.0)
    Z["Income_per_Age"] = X["Annual_Income_USD"] / (X["Age"] + 1.0)
    Z["Commute_per_Car"] = X["Daily_Commute_km"] / (X["Number_of_Cars_Owned"] + 1.0)
    Z["Income_per_Charging"] = X["Annual_Income_USD"] / (X["Total_Charging_Stations"] + 1.0)
    return Z


def add_triple_features(X: pd.DataFrame) -> pd.DataFrame:
    """三阶交互特征 (无泄漏)。"""
    Z = X.copy()
    Z["Env_Income_Subsidy"] = X["Environmental_Concern_Level"] * X["Annual_Income_USD"] * X["Subsidy_Available"]
    Z["Commute_Income_Anxiety"] = X["Daily_Commute_km"] * X["Annual_Income_USD"] * X["Range_Anxiety_Level"]
    Z["Cars_Charging_Env"] = X["Number_of_Cars_Owned"] * X["Home_Charging_Possible"] * X["Environmental_Concern_Level"]
    return Z


def add_bin_features(X: pd.DataFrame) -> pd.DataFrame:
    """分箱特征 (无泄漏)。"""
    Z = X.copy()
    Z["Income_bin10"] = pd.qcut(X["Annual_Income_USD"], 10, labels=False, duplicates="drop").astype("category")
    Z["Age_bin"] = pd.cut(X["Age"], bins=[24, 34, 44, 54, 70], labels=False).astype("category")
    Z["Commute_bin"] = pd.qcut(X["Daily_Commute_km"], 5, labels=False, duplicates="drop").astype("category")
    return Z


def run_cv(X, y, X_test, extra_cat=()):
    """单 LGB 5 折, 返回 OOF AUC。X 已含所有特征。"""
    cats = [c for c in CAT_COLS if c in X.columns] + [c for c in extra_cat if c in X.columns]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        d_tr = lgb.Dataset(X.iloc[tr], y.iloc[tr], categorical_feature=cats)
        d_va = lgb.Dataset(X.iloc[va], y.iloc[va], reference=d_tr)
        m = lgb.train(lgb_params(), d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        del d_tr, d_va, m; gc.collect()
    return roc_auc_score(y, oof)


def run_cv_with_te(X, y, X_test, te_cols, smoothing=20.0):
    """带目标编码(防泄漏, fold 内平滑编码)的单 LGB 5 折。"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    cats = [c for c in CAT_COLS if c in X.columns]
    for tr, va in skf.split(X, y):
        X_tr = X.iloc[tr].copy()
        X_va = X.iloc[va].copy()
        y_tr = y.iloc[tr]
        global_mean = y_tr.mean()
        for c in te_cols:
            agg = y_tr.groupby(X_tr[c]).agg(["mean", "count"])
            enc = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
            X_tr[f"TE_{c}"] = X_tr[c].map(enc).fillna(global_mean).astype(float)
            X_va[f"TE_{c}"] = X_va[c].map(enc).fillna(global_mean).astype(float)
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
    X_test = build_features(test)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    log(f"[data] X {X.shape} | y+ {y.mean():.4f}\n")

    # A. baseline
    log(f"[A] baseline              = {run_cv(X, y, X_test):.5f}")

    # B. + 比率
    Xb = add_ratio_features(X)
    log(f"[B] + ratio               = {run_cv(Xb, y, X_test):.5f}")
    del Xb; gc.collect()

    # C. + 三阶
    Xc = add_triple_features(X)
    log(f"[C] + triple              = {run_cv(Xc, y, X_test):.5f}")
    del Xc; gc.collect()

    # D. + 分箱
    Xd = add_bin_features(X)
    log(f"[D] + bins                = {run_cv(Xd, y, X_test, extra_cat=['Income_bin10','Age_bin','Commute_bin']):.5f}")
    del Xd; gc.collect()

    # E. + 目标编码 (3 个 nominal category)
    te_cols = NOMINAL_CATS  # Gender, City_Type, Current_Car_Type
    log(f"[E] + target-encode      = {run_cv_with_te(X, y, X_test, te_cols):.5f}")

    log(f"\n[total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
