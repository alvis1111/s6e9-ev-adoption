"""共享：Naji 特征工程（数字位/原始均值/频率/魔法/平滑分箱）。TE 在调用方折内做。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
TARGET = "Will_Buy_EV"


def build_features(train_df=None, test_df=None):
    train = pd.read_csv(BASE / "train.csv") if train_df is None else train_df.copy()
    test = pd.read_csv(BASE / "test.csv") if test_df is None else test_df.copy()
    orig = pd.read_csv(BASE / "EV_orig.csv")
    train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0})
    orig[TARGET] = orig[TARGET].map({"Yes": 1, "No": 0})
    train["is_train"] = 1
    test["is_train"] = 0
    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])
    test[TARGET] = np.nan
    combined = pd.concat([train, test], ignore_index=True)
    combined.drop(columns=["Number_of_Cars_Owned"], inplace=True, errors="ignore")
    cat_cols = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols + ["id", "is_train", TARGET]]

    digit_features = []
    for c in num_cols:
        for k in range(-4, 4):
            cn = f"{c}_digit{k}"
            combined[cn] = (combined[c].fillna(0) // (10**k) % 10).astype("int8")
            digit_features.append(cn)
    num_cols.extend(digit_features)

    ogm = orig[TARGET].mean()
    for col in cat_cols + num_cols:
        if col in orig.columns:
            stats = orig.groupby(col, observed=False)[TARGET].mean()
            combined[f"{col}_org_mean"] = combined[col].map(stats).fillna(ogm).astype(float)

    num_to_cat_cols = []
    for col in num_cols:
        cn = f"{col}_cat"
        combined[cn] = combined[col].fillna("NaN").astype(str)
        num_to_cat_cols.append(cn)

    all_cats = cat_cols + num_to_cat_cols
    for col in all_cats:
        fm = combined[col].value_counts(normalize=True).to_dict()
        combined[f"{col}_fe"] = combined[col].map(fm).astype(float).fillna(0.0)

    combined["is_30k_spike"] = (combined["Annual_Income_USD"] == 30000.0).astype("int8")
    combined["is_millionaire_cliff"] = (combined["Annual_Income_USD"] >= 170537.0).astype("int8")
    combined["is_dead_zone"] = ((combined["Annual_Income_USD"] >= 38000.0) & (combined["Annual_Income_USD"] <= 42000.0)).astype("int8")
    combined["is_env_hater"] = (combined["Environmental_Concern_Level"] == 1).astype("int8")

    combined["income_exact_int"] = np.floor(combined["Annual_Income_USD"]).astype(str)
    combined["income100_floor"] = np.floor(combined["Annual_Income_USD"] / 100.0).astype(str)
    combined["income1000_floor"] = np.floor(combined["Annual_Income_USD"] / 1000.0).astype(str)
    combined["commute_integer"] = np.floor(combined["Daily_Commute_km"]).astype(str)
    all_cats.extend(["income_exact_int", "income100_floor", "income1000_floor", "commute_integer"])

    train = combined[combined["is_train"] == 1].drop(columns=["is_train"]).copy()
    test = combined[combined["is_train"] == 0].drop(columns=["is_train", TARGET]).copy()

    eval_cols = [c for c in train.columns if c not in ["id", TARGET] and pd.api.types.is_numeric_dtype(train[c])]
    corr = train[eval_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if any(upper[c] == 1.0)]
    drop_const = [c for c in train.columns if train[c].nunique() == 1] + [c for c in test.columns if test[c].nunique() == 1]
    DROP = [c for c in set(drop_corr).union(set(drop_const)) if c not in ["id", TARGET]]
    train.drop(columns=DROP, inplace=True, errors="ignore")
    test.drop(columns=DROP, inplace=True, errors="ignore")

    FEATURES = [c for c in test.columns if c != "id"]
    TE_COLS = [c for c in all_cats if c not in DROP]
    return train[FEATURES], train[TARGET], test[FEATURES], test["id"], TE_COLS


def make_lgb(seed: int):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=20000, learning_rate=0.02, max_depth=5, num_leaves=32,
        min_child_samples=10, subsample=0.812763, colsample_bytree=0.30293,
        reg_alpha=0.07094, reg_lambda=2.03303, max_bin=1024, random_state=seed,
        feature_pre_filter=False, metric="auc", n_jobs=-1, verbose=-1,
    )


def te_transform(X_tr, y_tr, X_va, X_te, TE_COLS):
    """三套目标编码（auto/10/100），返回变换后（原地修改）的三个 DataFrame。"""
    from sklearn.preprocessing import TargetEncoder

    encs = {}
    for tag, smooth in [("auto", "auto"), ("10", 10.0), ("100", 100.0)]:
        enc = TargetEncoder(shuffle=True, cv=5, smooth=smooth, random_state=42)
        encs[f"tr_{tag}"] = enc.fit_transform(X_tr[TE_COLS], y_tr)
        encs[f"va_{tag}"] = enc.transform(X_va[TE_COLS])
        encs[f"te_{tag}"] = enc.transform(X_te[TE_COLS])
    for i, col in enumerate(TE_COLS):
        for tag in ["auto", "10", "100"]:
            X_tr[f"{col}_TE_{tag}"] = encs[f"tr_{tag}"][:, i].astype("float32")
            X_va[f"{col}_TE_{tag}"] = encs[f"va_{tag}"][:, i].astype("float32")
            X_te[f"{col}_TE_{tag}"] = encs[f"te_{tag}"][:, i].astype("float32")
        X_tr.drop(columns=[col], inplace=True)
        X_va.drop(columns=[col], inplace=True)
        X_te.drop(columns=[col], inplace=True)
    return X_tr, X_va, X_te
