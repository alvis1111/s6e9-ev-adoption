"""
EV Purchases — Playground Series S6E9
特征工程实验 (Feature Engineering)
================================================
在 baseline 之上新增组合/衍生特征，用同样的分层 5-Fold CV 评估，
与 baseline OOF AUC (0.94166) 对比，判断特征是否有效。

运行:  python ev_features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from ev_clean import load_clean_data, FEATURE_COLS, TARGET

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5

LGB_CATEGORY_COLS = ["Gender", "City_Type", "Current_Car_Type"]
ORDINAL_MAP = {"Low": 0, "Medium": 1, "High": 2}
BASELINE_AUC = 0.94166


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """原始列 + 衍生特征，返回数值/分类矩阵。"""
    X = df[FEATURE_COLS].copy()

    # 有序分类 -> 整数
    X["Range_Anxiety_Level"] = X["Range_Anxiety_Level"].map(ORDINAL_MAP).astype("int8")

    # 名义分类 -> category (LightGBM 原生)
    for c in LGB_CATEGORY_COLS:
        X[c] = X[c].astype("category")

    # ---- 衍生特征 ----
    # 1. 充电可达性
    X["Total_Charging_Stations"] = (
        X["Charging_Stations_Near_Home"] + X["Charging_Stations_Near_Work"]
    )
    X["Home_Charging_Access"] = (
        X["Home_Charging_Possible"] * X["Charging_Stations_Near_Home"]
    )

    # 2. 收入维度
    X["Income_per_Car"] = X["Annual_Income_USD"] / X["Number_of_Cars_Owned"]
    X["Income_per_km"] = X["Annual_Income_USD"] / (X["Daily_Commute_km"] + 1.0)

    # 3. 关键交互 (top 特征之间)
    X["Env_x_Income"] = X["Environmental_Concern_Level"] * X["Annual_Income_USD"]
    X["Env_x_Subsidy"] = X["Environmental_Concern_Level"] * X["Subsidy_Available"]
    X["Income_x_Subsidy"] = X["Annual_Income_USD"] * X["Subsidy_Available"]
    X["Commute_x_Anxiety"] = X["Daily_Commute_km"] * X["Range_Anxiety_Level"]
    X["Env_x_Anxiety"] = X["Environmental_Concern_Level"] * X["Range_Anxiety_Level"]

    # 4. 收入分箱 (作为 category，捕捉非线性)
    X["Income_bin"] = pd.qcut(X["Annual_Income_USD"], 5, labels=False, duplicates="drop").astype("category")

    # 5. Age 平方 (非单调可能性)
    X["Age_sq"] = X["Age"] ** 2

    return X


def main() -> None:
    train, test, sub = load_clean_data(DATA_DIR)

    X = build_features(train)
    y = train[TARGET]
    X_test = build_features(test)

    print(f"[data] X {X.shape} | y {y.shape} | X_test {X_test.shape}")
    print(f"[target] positive rate = {y.mean():.4f}\n")

    cat_cols = LGB_CATEGORY_COLS + ["Income_bin"]

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    gain = np.zeros(len(X.columns))

    print("[CV] stratified 5-fold ...")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        model = lgb.train(
            params, d_tr, num_boost_round=3000,
            valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        gain += model.feature_importance("gain") / N_FOLDS

        auc = roc_auc_score(y_va, oof[va_idx])
        print(f"  fold {fold}: AUC = {auc:.5f}")

    oof_auc = roc_auc_score(y, oof)
    print(f"\n[OOF] mean AUC = {oof_auc:.5f}  (baseline {BASELINE_AUC}, delta {oof_auc - BASELINE_AUC:+.5f})")

    imp = pd.Series(gain, index=X.columns).sort_values(ascending=False)
    print("\n[feature importance (gain, avg)]")
    print(imp.round(1).to_string())

    # 保存提交 (若提升则替换)
    sub["Will_Buy_EV"] = test_pred
    out = DATA_DIR / "submission_features.csv"
    sub.to_csv(out, index=False)
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
