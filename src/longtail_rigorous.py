"""S6E9 —— 严谨流程复测两个长尾（聚类特征 / 交互特征）。

在 rigorous 框架（10% 审计留出 + 固定 5 折）下，对三个变体逐个跑同一组折：
    baseline（Naji 原样）/ +聚类(k=1000) / +交互(收入十分位×关键类别)
比较每个变体的 dev OOF、成对折 delta、audit AUC。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from naji_features import build_features, make_lgb, te_transform

BASE = Path(__file__).parent
TARGET = "Will_Buy_EV"
INTERACT = ["Environmental_Concern_Level", "Subsidy_Available", "Range_Anxiety_Level", "Home_Charging_Possible"]


def add_cluster(X, X_test):
    num_cols = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level"]
    cat_cols = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
    raw = pd.concat([X[num_cols].astype(float), pd.get_dummies(X[cat_cols])], axis=1)
    raw_te = pd.concat([X_test[num_cols].astype(float), pd.get_dummies(X_test[cat_cols])], axis=1)
    scaler = StandardScaler().fit(raw)
    km = MiniBatchKMeans(n_clusters=1000, random_state=42, batch_size=20000, n_init=3).fit(scaler.transform(raw))
    X = X.copy(); X_test = X_test.copy()
    X["_cluster"] = km.predict(scaler.transform(raw)).astype(str)
    X_test["_cluster"] = km.predict(scaler.transform(raw_te)).astype(str)
    return X, X_test, ["_cluster"]


def add_interaction(X, X_test):
    inc = pd.concat([X["Annual_Income_USD"], X_test["Annual_Income_USD"]])
    q = pd.qcut(inc, 10, labels=False, duplicates="drop")
    X = X.copy(); X_test = X_test.copy()
    X["_inc_dec"] = q.iloc[: len(X)].astype(str)
    X_test["_inc_dec"] = q.iloc[len(X):].astype(str)
    new_cats = []
    for c in INTERACT:
        nm = f"_inc_{c}"
        X[nm] = X["_inc_dec"] + "_" + X[c].astype(str)
        X_test[nm] = X_test["_inc_dec"] + "_" + X_test[c].astype(str)
        new_cats.append(nm)
    X.drop(columns=["_inc_dec"], inplace=True)
    X_test.drop(columns=["_inc_dec"], inplace=True)
    return X, X_test, new_cats


def run_variant(X, y, X_test, TE_COLS, n_audit, audit_y, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    te_pred = np.zeros(len(X_test))
    fold_aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx].copy(), y.iloc[va_idx]
        X_te = X_test.copy()
        X_tr, X_va, X_te = te_transform(X_tr, y_tr, X_va, X_te, TE_COLS)
        clf = make_lgb(seed)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(500, verbose=False)])
        oof[va_idx] = clf.predict_proba(X_va)[:, 1]
        te_pred += clf.predict_proba(X_te)[:, 1] / 5
        fold_aucs.append(roc_auc_score(y_va, oof[va_idx]))
    dev_auc = roc_auc_score(y, oof)
    audit_auc = roc_auc_score(audit_y, te_pred[:n_audit])
    return fold_aucs, dev_auc, audit_auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed = args.seed
    t0 = time.time()

    tr = pd.read_csv(BASE / "train.csv")
    y = (tr[TARGET] == "Yes").astype(int).values
    dev_idx, audit_idx = train_test_split(np.arange(len(tr)), test_size=0.10, stratify=y, random_state=42)
    dev = tr.iloc[dev_idx].reset_index(drop=True)
    audit = tr.iloc[audit_idx].reset_index(drop=True)
    audit_y = (audit[TARGET] == "Yes").astype(int).values
    test_df = pd.concat([audit, pd.read_csv(BASE / "test.csv")], ignore_index=True)
    n_audit = len(audit)

    X, y_dev, X_test, ids, TE_COLS = build_features(train_df=dev, test_df=test_df)

    variants = {
        "baseline": (X, X_test, []),
        "cluster": add_cluster(X, X_test),
        "interaction": add_interaction(X, X_test),
    }
    results = {}
    for name, (Xv, Xtv, extra_cats) in variants.items():
        TE = TE_COLS + extra_cats
        fold_aucs, dev_auc, audit_auc = run_variant(Xv, y_dev, Xtv, TE, n_audit, audit_y, seed)
        results[name] = (fold_aucs, dev_auc, audit_auc)
        print(f"  {name:<12} dev OOF {dev_auc:.5f} | audit {audit_auc:.5f} | 折 {[f'{a:.5f}' for a in fold_aucs]}  [{time.time()-t0:.0f}s]", flush=True)

    print("\n=== 成对折 delta（vs baseline，正=涨）===")
    base_folds = results["baseline"][0]
    for name in ["cluster", "interaction"]:
        deltas = [c - b for b, c in zip(base_folds, results[name][0])]
        print(f"  {name}: delta {[f'{d:+.5f}' for d in deltas]} | 均值 {np.mean(deltas):+.5f} | 涨折数 {sum(d>0 for d in deltas)}/5")


if __name__ == "__main__":
    main()
